from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, TYPE_CHECKING

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils._pytree import tree_map

from transformers import PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from ttt.triton_linear import TritonLinear
from ttt.triton_mlp_split import TritonMLPSplit

@dataclass
class ModelArgs:
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: Optional[int] = None

    ffn_intermediate_dim: int = 2048 # Directly specify SwiGlu hidden dimension
    tie_word_embeddings: bool = False  # Add to match JAX
    vocab_size: int = -1

    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-6
    rope_theta: float = 10000

    max_seq_len: int = 2048 # Used for RoPE

    # If `True`, then each transformer block init uses its layer ID, and if
    # `False`, each uses the total number of transformer blocks
    depth_init: bool = True
    
    # TTT-Config
    seq_modeling_block: str = "self_attention"
    ttt_base_lr: float = 1.0
    mini_batch_size: int = 16
    scan_checkpoint_group_size: int = 4

    # Match JAX initializer
    initializer_range: float = 0.02
    fix_normal_initializer_range: bool = False
    
########################
### Backbone Modules ###
########################

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    The input freqs_cis tensor is assumed to be of shape (max_seqlen, dim),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

#########################
### TTT Layer Modules ###
#########################


def scan(f, init, xs, checkpoint_group=0):
    """Minic jax.lax.scan function."""
    carry = init
    if isinstance(xs, dict):
        num_items = len(next(iter(xs.values())))
    else:
        num_items = len(xs[0])

    def scan_fn(carry, i_start, i_end):
        sub_out_list = []
        for i in range(i_start, i_end):
            if isinstance(xs, dict):
                x = {key: tensor[i] for key, tensor in xs.items()}
            else:
                x = [x[i] for x in xs]
            carry, y = f(carry, x)
            sub_out_list.append(y)
        sub_out = torch.stack(sub_out_list)
        return carry, sub_out

    if checkpoint_group > 0:
        out_list = []
        for k in range(0, num_items, checkpoint_group):
            carry, sub_out = torch.utils.checkpoint.checkpoint(
                scan_fn, carry, k, min(k + checkpoint_group, num_items), use_reentrant=False
            )
            out_list.append(sub_out)
        out = torch.concatenate(out_list, dim=0)
    else:
        carry, out = scan_fn(carry, 0, num_items)

    return carry, out


def ln_fwd(x, gamma, beta, eps=1e-6):
    "Batch forward for LayerNorm."

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    # Scale and shift
    y = gamma * x_hat + beta

    return y


def ln_fused_l2_bwd(x, l2_target, gamma, beta, eps=1e-6):
    "Batch backward for LayerNorm fused with L2 loss."
    D = x.shape[-1]

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    # Scale and shift
    y = gamma * x_hat + beta

    grad_output = y - l2_target
    grad_x_hat = grad_output * gamma
    z = (
        (1.0 / D)
        * (
            D * grad_x_hat
            - grad_x_hat.sum(dim=-1, keepdim=True)
            - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
        )
        / std
    )

    return z


def gelu_bwd(x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff


class TTTCache:
    pass


class TTTBase(nn.Module):
    def __init__(self, config: 'ModelArgs'):
        super().__init__()
        self.config = config
        self.width = config.dim
        self.num_heads = config.n_heads
        self.num_kv_heads = config.n_kv_heads if config.n_kv_heads is not None else config.n_heads
        self.head_dim = self.width // self.num_heads
        self.mini_batch_size = config.mini_batch_size

        token_idx = 1.0 / torch.arange(1, self.mini_batch_size + 1)
        self.register_buffer("token_idx", token_idx, persistent=False)
        self.learnable_token_idx = nn.Parameter(torch.zeros((self.mini_batch_size,)))

        self._init_qkvo_proj()

        self._init_ttt_lr_gate()
        self._init_ttt_ln()

        self.post_norm = nn.LayerNorm(self.width, eps=1e-6)

    def init_weights(self, init_std: float):
        if self.config.fix_normal_initializer_range:
            for linear in (self.wq, self.wk, self.wv):
                nn.init.normal_(linear.weight, mean=0.0, std=self.config.initializer_range)
            nn.init.normal_(self.wo.weight, mean=0.0, std=self.config.initializer_range)
        else:
            for linear in (self.wq, self.wk, self.wv):
                nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
            nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

        # @xinhao: must explicitly initialize, otherwise become 0 after transferring from meta device to real device
        self.token_idx.copy_(1.0 / torch.arange(1, self.mini_batch_size + 1))
        self.post_norm.reset_parameters()
        self.ttt_norm_weight.data.copy_(torch.ones_like(self.ttt_norm_weight.data))
        self.ttt_norm_bias.data.copy_(torch.zeros_like(self.ttt_norm_bias.data))
        self.learnable_ttt_lr_weight.data.copy_(torch.randn_like(self.learnable_ttt_lr_weight.data) * 0.02)
        self.learnable_ttt_lr_bias.data.copy_(torch.zeros_like(self.learnable_ttt_lr_bias.data))

    def _init_qkvo_proj(self):
        self.wq = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.width, self.num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.width, self.num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)

    def _init_ttt_lr_gate(self):
        linear_weight_data = nn.Linear(self.width, 1, bias=True).weight.data
        self.learnable_ttt_lr_weight = nn.Parameter(
            torch.stack(
                [torch.normal(0, 0.02, size=linear_weight_data.shape) for _ in range(self.num_heads)],
                dim=0,
            )
        )

        linear_bias_data = nn.Linear(self.width, 1, bias=True).bias.data
        self.learnable_ttt_lr_bias = nn.Parameter(
            torch.stack(
                [torch.zeros_like(linear_bias_data) for _ in range(self.num_heads)],
                dim=0,
            )
        )

    def _init_ttt_ln(self):
        ln_weight_data = nn.LayerNorm(self.head_dim).weight.data

        self.ttt_norm_weight = nn.Parameter(torch.tile(ln_weight_data.unsqueeze(0), (self.num_heads, 1)))
        ln_bias_data = nn.LayerNorm(self.head_dim).bias.data
        self.ttt_norm_bias = nn.Parameter(torch.tile(ln_bias_data.unsqueeze(0), (self.num_heads, 1)))

    def get_qkv_projections(self, hidden_states):
        XQ, XK, XV = (
            self.wq(hidden_states),
            self.wk(hidden_states),
            self.wv(hidden_states),
        )
        return XQ, XK, XV

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def get_eta(self, X, mini_batch_step_offset, mini_batch_size):
        ttt_lr = torch.einsum("bnkc,hdc->bhnkd", X, self.learnable_ttt_lr_weight) + self.learnable_ttt_lr_bias.reshape(
            1, -1, 1, 1, 1
        )
        ttt_lr = F.sigmoid(ttt_lr)

        ttt_lr = ttt_lr.permute(0, 1, 2, 4, 3)
        ttt_lr_eta = self.config.ttt_base_lr * ttt_lr / self.head_dim

        token_idx = self.token_idx + self.learnable_token_idx
        token_idx = token_idx[mini_batch_step_offset : mini_batch_step_offset + mini_batch_size]

        token_idx = torch.clamp_min(
            token_idx, 0.0
        )  # TODO: this can lead to a dead lock where no time-mixing can happen

        token_eta = torch.broadcast_to(
            token_idx.reshape(1, 1, 1, mini_batch_size, 1),
            (X.shape[0], self.num_heads, X.shape[1], mini_batch_size, 1),
        )

        return token_eta, ttt_lr_eta

    def get_ttt_inputs(self, inputs, mini_batch_size):
        XQ = inputs["XQ"]
        XK = inputs["XK"]
        XV = inputs["XV"]
        X = inputs["X"]
        B, L, C = X.shape
        num_mini_batch = L // mini_batch_size

        X = X.reshape(B, num_mini_batch, mini_batch_size, self.width)

        XQ = XQ.reshape(B, self.num_heads, L // mini_batch_size, mini_batch_size, self.head_dim)
        XK = XK.reshape(B, self.num_heads, L // mini_batch_size, mini_batch_size, self.head_dim)
        XV = XV.reshape(B, self.num_heads, L // mini_batch_size, mini_batch_size, self.head_dim)

        mini_batch_step_offset = 0
        token_eta, ttt_lr_eta = self.get_eta(X, mini_batch_step_offset, mini_batch_size)
        eta = token_eta * ttt_lr_eta

        inputs = {
            "XQ": XQ,
            "XK": XK,
            "XV": XV,
            "eta": eta,
            "token_eta": token_eta,
            "ttt_lr_eta": ttt_lr_eta,
        }
        return inputs

    def ttt(
        self,
        inputs,
    ):
        raise NotImplementedError("ttt method must be implemented in TTTBase subclasses.")

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        cache_params: Optional[TTTCache] = None,
    ):
        assert cache_params is None, "TTT mini doesn't support cache_params."

        B, L = hidden_states.shape[:2]

        XQ, XK, XV = self.get_qkv_projections(hidden_states)

        XQ = XQ.view(B, L, -1, self.head_dim)
        XK = XK.view(B, L, -1, self.head_dim)
        XV = XV.view(B, L, -1, self.head_dim)

        freqs_cis = freqs_cis[:self.mini_batch_size].tile(L // self.mini_batch_size, 1)
        
        XQ, XK = apply_rotary_emb(XQ, XK, freqs_cis=freqs_cis)
        
        XQ, XK, XV = XQ.transpose(1, 2), XK.transpose(1, 2), XV.transpose(1, 2)
        
        # GQA
        XK = XK.tile(1, self.num_heads // self.num_kv_heads, 1, 1)
        XV = XV.tile(1, self.num_heads // self.num_kv_heads, 1, 1)
        
        inputs = {
            "XQ": XQ,
            "XK": XK,
            "XV": XV,
            "X": hidden_states,
        }

        output_hidden_states, _ = self.ttt(
            self.get_ttt_inputs(inputs, self.mini_batch_size),
        )

        output_hidden_states = self.post_norm(output_hidden_states)
        output_hidden_states = self.wo(output_hidden_states)

        return output_hidden_states


class TTTLinear(TTTBase):
    def __init__(self, config):
        super().__init__(config)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

    def init_weights(self, init_std: float):
        if self.config.fix_normal_initializer_range:
            for linear in (self.wq, self.wk, self.wv):
                nn.init.normal_(linear.weight, mean=0.0, std=self.config.initializer_range)
            nn.init.normal_(self.wo.weight, mean=0.0, std=self.config.initializer_range)
        else:
            for linear in (self.wq, self.wk, self.wv):
                nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
            nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

        # @xinhao: must explicitly initialize, otherwise become 0 after transferring from meta device to real
        self.token_idx.copy_(1.0 / torch.arange(1, self.mini_batch_size + 1))
        self.post_norm.reset_parameters()
        self.ttt_norm_weight.data.copy_(torch.ones_like(self.ttt_norm_weight.data))
        self.ttt_norm_bias.data.copy_(torch.zeros_like(self.ttt_norm_bias.data))
        self.learnable_ttt_lr_weight.data.copy_(torch.randn_like(self.learnable_ttt_lr_weight.data) * 0.02)
        self.learnable_ttt_lr_bias.data.copy_(torch.zeros_like(self.learnable_ttt_lr_bias.data))
        self.W1.data.copy_(torch.randn_like(self.W1.data) * 0.02)
        self.b1.data.copy_(torch.zeros_like(self.b1.data))

    def ttt(self, inputs, use_dual_form=True):
        mini_batch_size = self.mini_batch_size

        B = inputs["XV"].shape[0]
        num_mini_batch = inputs["XV"].shape[2]
        L = inputs["XV"].shape[2] * inputs["XV"].shape[3]
        device = inputs["XV"].device
        dtype = inputs["XV"].dtype

        def compute_mini_batch(params_dict, inputs):
            W1_init = params_dict["W1_states"]
            b1_init = params_dict["b1_states"]

            XQ_mini_batch = inputs["XQ"]
            XV_mini_batch = inputs["XV"]
            XK_mini_batch = inputs["XK"]

            eta_mini_batch = inputs["eta"]
            token_eta_mini_batch = inputs["token_eta"]
            ttt_lr_eta_mini_batch = inputs["ttt_lr_eta"]

            X1 = XK_mini_batch
            Z1 = X1 @ W1_init + b1_init
            reconstruction_target = XV_mini_batch - XK_mini_batch

            ln_weight = self.ttt_norm_weight.reshape(self.num_heads, 1, self.head_dim)
            ln_bias = self.ttt_norm_bias.reshape(self.num_heads, 1, self.head_dim)
            grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias)

            if use_dual_form:
                Attn1 = torch.tril(XQ_mini_batch @ X1.transpose(-2, -1))
                b1_bar = b1_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z1
                Z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar

                last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
                W1_last = W1_init - (last_eta_mini_batch * X1).transpose(-1, -2) @ grad_l_wrt_Z1
                b1_last = b1_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z1, dim=-2, keepdim=True)
            else:
                ttt_lr_eta_mini_batch = torch.broadcast_to(
                    ttt_lr_eta_mini_batch,
                    (
                        *ttt_lr_eta_mini_batch.shape[:2],
                        mini_batch_size,
                        mini_batch_size,
                    ),
                )

                grad_W1 = torch.einsum("bhki,bhkj->bhkij", X1, grad_l_wrt_Z1)
                grad_W1 = torch.einsum("bhnk,bhkij->bhnij", torch.tril(ttt_lr_eta_mini_batch), grad_W1)
                grad_b1 = torch.einsum("bhnk,bhki->bhni", torch.tril(ttt_lr_eta_mini_batch), grad_l_wrt_Z1)

                W1_bar = W1_init.unsqueeze(2) - grad_W1 * token_eta_mini_batch.unsqueeze(-1)
                b1_bar = b1_init - grad_b1 * token_eta_mini_batch

                Z1_bar = (XQ_mini_batch.unsqueeze(3) @ W1_bar).squeeze(3) + b1_bar

                W1_last = W1_bar[:, :, -1]
                b1_last = b1_bar[:, :, -1:]

            Z1_bar = ln_fwd(Z1_bar, ln_weight, ln_bias)

            XQW_mini_batch = XQ_mini_batch + Z1_bar

            last_param_dict = {
                "W1_states": W1_last,
                "b1_states": b1_last,
            }
            return last_param_dict, XQW_mini_batch

        init_params_dict = {
            "W1_states": torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1)),
            "b1_states": torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1)),
        }

        inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)

        XQW_batch = torch.empty(
            (num_mini_batch, B, self.num_heads, mini_batch_size, self.head_dim),
            device=device,
            dtype=dtype,
        )

        batch_params_dict, XQW_batch = scan(
            compute_mini_batch,
            init_params_dict,
            inputs,
            self.config.scan_checkpoint_group_size if self.training else 0,
        )

        XQW_batch = XQW_batch.permute(1, 0, 3, 2, 4)
        XQW_batch = XQW_batch.reshape(B, L, self.width)
        return XQW_batch, batch_params_dict


class TTTLinearTriton(TTTBase):
    def __init__(self, config: 'ModelArgs'):
        super().__init__(config)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

    def init_weights(self, init_std: float):
        if self.config.fix_normal_initializer_range:
            for linear in (self.wq, self.wk, self.wv):
                nn.init.normal_(linear.weight, mean=0.0, std=self.config.initializer_range)
            nn.init.normal_(self.wo.weight, mean=0.0, std=self.config.initializer_range)
        else:
            for linear in (self.wq, self.wk, self.wv):
                nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
            nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

        # @xinhao: must explicitly initialize, otherwise become 0 after transferring from meta device to real device
        self.token_idx.copy_(1.0 / torch.arange(1, self.mini_batch_size + 1))
        self.post_norm.reset_parameters()
        self.ttt_norm_weight.data.copy_(torch.ones_like(self.ttt_norm_weight.data))
        self.ttt_norm_bias.data.copy_(torch.zeros_like(self.ttt_norm_bias.data))
        self.learnable_ttt_lr_weight.data.copy_(torch.randn_like(self.learnable_ttt_lr_weight.data) * 0.02)
        self.learnable_ttt_lr_bias.data.copy_(torch.zeros_like(self.learnable_ttt_lr_bias.data))
        self.W1.data.copy_(torch.randn_like(self.W1.data) * 0.02)
        self.b1.data.copy_(torch.zeros_like(self.b1.data))

    def ttt(self, inputs, use_dual_form=True):
        mini_batch_size = self.mini_batch_size

        B = inputs["XV"].shape[0]
        num_mini_batch = inputs["XV"].shape[2]
        L = inputs["XV"].shape[2] * inputs["XV"].shape[3]
        device = inputs["XV"].device
        dtype = inputs["XV"].dtype

        W1_states = torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1))
        b1_states = torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1))

        checkpoint_group_size = (
            self.config.scan_checkpoint_group_size if self.config.scan_checkpoint_group_size > 0 else num_mini_batch
        )

        W1_last, b1_last, XQW_batch = TritonLinear.apply(
            self.ttt_norm_weight,
            self.ttt_norm_bias,
            W1_states,
            b1_states,
            inputs["XQ"],
            inputs["XV"],
            inputs["XK"],
            inputs["eta"],
            checkpoint_group_size,
        )

        batch_params_dict = {
            "W1_states": W1_last,
            "b1_states": b1_last,
        }

        XQW_batch = XQW_batch.permute(0, 2, 3, 1, 4)
        XQW_batch = XQW_batch.reshape(B, L, self.width)
        return XQW_batch, batch_params_dict


class TTTMLP(TTTBase):
    def __init__(self, config: 'ModelArgs'):
        super().__init__(config)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, 4 * self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, 4 * self.head_dim))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 4 * self.head_dim, self.head_dim)))
        self.b2 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

    def init_weights(self, init_std: float):
        if self.config.fix_normal_initializer_range:
            for linear in (self.wq, self.wk, self.wv):
                nn.init.normal_(linear.weight, mean=0.0, std=self.config.initializer_range)
            nn.init.normal_(self.wo.weight, mean=0.0, std=self.config.initializer_range)
        else:
            for linear in (self.wq, self.wk, self.wv):
                nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
            nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

        # @xinhao: must explicitly initialize, otherwise become 0 after transferring from meta device to real device
        self.token_idx.copy_(1.0 / torch.arange(1, self.mini_batch_size + 1))
        self.post_norm.reset_parameters()
        self.ttt_norm_weight.data.copy_(torch.ones_like(self.ttt_norm_weight.data))
        self.ttt_norm_bias.data.copy_(torch.zeros_like(self.ttt_norm_bias.data))
        self.learnable_ttt_lr_weight.data.copy_(torch.randn_like(self.learnable_ttt_lr_weight.data) * 0.02)
        self.learnable_ttt_lr_bias.data.copy_(torch.zeros_like(self.learnable_ttt_lr_bias.data))
        self.W1.data.copy_(torch.randn_like(self.W1.data) * 0.02)
        self.b1.data.copy_(torch.zeros_like(self.b1.data))
        self.W2.data.copy_(torch.randn_like(self.W2.data) * 0.02)
        self.b2.data.copy_(torch.zeros_like(self.b2.data))

    def ttt(self, inputs, use_dual_form=True):
        mini_batch_size = self.mini_batch_size

        B = inputs["XV"].shape[0]
        num_mini_batch = inputs["XV"].shape[2]
        L = inputs["XV"].shape[2] * inputs["XV"].shape[3]
        device = inputs["XV"].device
        dtype = inputs["XV"].dtype

        def compute_mini_batch(params_dict, inputs):
            W1_init = params_dict["W1_states"]
            b1_init = params_dict["b1_states"]
            W2_init = params_dict["W2_states"]
            b2_init = params_dict["b2_states"]

            XQ_mini_batch = inputs["XQ"]
            XV_mini_batch = inputs["XV"]
            XK_mini_batch = inputs["XK"]

            eta_mini_batch = inputs["eta"]
            token_eta_mini_batch = inputs["token_eta"]
            ttt_lr_eta_mini_batch = inputs["ttt_lr_eta"]

            X1 = XK_mini_batch
            Z1 = X1 @ W1_init + b1_init
            X2 = F.gelu(Z1, approximate="tanh")
            Z2 = X2 @ W2_init + b2_init
            reconstruction_target = XV_mini_batch - XK_mini_batch

            ln_weight = self.ttt_norm_weight.reshape(self.num_heads, 1, self.head_dim)
            ln_bias = self.ttt_norm_bias.reshape(self.num_heads, 1, self.head_dim)
            grad_l_wrt_Z2 = ln_fused_l2_bwd(Z2, reconstruction_target, ln_weight, ln_bias)
            grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2_init.transpose(-2, -1) * gelu_bwd(Z1)

            if use_dual_form:
                Attn1 = torch.tril(XQ_mini_batch @ X1.transpose(-2, -1))
                b1_bar = b1_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z1
                Z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar
                X2_bar = F.gelu(Z1_bar, approximate="tanh")

                Attn2 = torch.tril(X2_bar @ X2.transpose(-2, -1))
                b2_bar = b2_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z2
                Z2_bar = X2_bar @ W2_init - (eta_mini_batch * Attn2) @ grad_l_wrt_Z2 + b2_bar

                last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
                W1_last = W1_init - (last_eta_mini_batch * X1).transpose(-1, -2) @ grad_l_wrt_Z1
                b1_last = b1_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z1, dim=-2, keepdim=True)
                W2_last = W2_init - (last_eta_mini_batch * X2).transpose(-1, -2) @ grad_l_wrt_Z2
                b2_last = b2_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z2, dim=-2, keepdim=True)

            else:
                ttt_lr_eta_mini_batch = torch.broadcast_to(
                    ttt_lr_eta_mini_batch,
                    (
                        *ttt_lr_eta_mini_batch.shape[:2],
                        mini_batch_size,
                        mini_batch_size,
                    ),
                )

                grad_W2 = torch.einsum("bhki,bhkj->bhkij", X2, grad_l_wrt_Z2)
                grad_W2 = torch.einsum("bhnk,bhkij->bhnij", torch.tril(ttt_lr_eta_mini_batch), grad_W2)
                grad_b2 = torch.einsum("bhnk,bhki->bhni", torch.tril(ttt_lr_eta_mini_batch), grad_l_wrt_Z2)

                grad_W1 = torch.einsum("bhki,bhkj->bhkij", X1, grad_l_wrt_Z1)
                grad_W1 = torch.einsum("bhnk,bhkij->bhnij", torch.tril(ttt_lr_eta_mini_batch), grad_W1)
                grad_b1 = torch.einsum("bhnk,bhki->bhni", torch.tril(ttt_lr_eta_mini_batch), grad_l_wrt_Z1)

                W1_bar = W1_init.unsqueeze(2) - grad_W1 * token_eta_mini_batch.unsqueeze(-1)
                b1_bar = b1_init - grad_b1 * token_eta_mini_batch
                W2_bar = W2_init.unsqueeze(2) - grad_W2 * token_eta_mini_batch.unsqueeze(-1)
                b2_bar = b2_init - grad_b2 * token_eta_mini_batch

                Z1_bar = (XQ_mini_batch.unsqueeze(3) @ W1_bar).squeeze(3) + b1_bar
                X2_bar = F.gelu(Z1_bar, approximate="tanh")
                Z2_bar = (X2_bar.unsqueeze(3) @ W2_bar).squeeze(3) + b2_bar

                W1_last = W1_bar[:, :, -1]
                b1_last = b1_bar[:, :, -1:]
                W2_last = W2_bar[:, :, -1]
                b2_last = b2_bar[:, :, -1:]

            Z2_bar = ln_fwd(Z2_bar, ln_weight, ln_bias)

            XQW_mini_batch = XQ_mini_batch + Z2_bar

            last_param_dict = {
                "W1_states": W1_last,
                "b1_states": b1_last,
                "W2_states": W2_last,
                "b2_states": b2_last,
            }
            return last_param_dict, XQW_mini_batch

        init_params_dict = {
            "W1_states": torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1)),
            "b1_states": torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1)),
            "W2_states": torch.tile(self.W2.unsqueeze(0), dims=(B, 1, 1, 1)),
            "b2_states": torch.tile(self.b2.unsqueeze(0), dims=(B, 1, 1, 1)),
        }

        inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)

        XQW_batch = torch.empty(
            (num_mini_batch, B, self.num_heads, mini_batch_size, self.head_dim),
            device=device,
            dtype=dtype,
        )

        batch_params_dict, XQW_batch = scan(
            compute_mini_batch,
            init_params_dict,
            inputs,
            self.config.scan_checkpoint_group_size if self.training else 0,
        )

        XQW_batch = XQW_batch.permute(1, 0, 3, 2, 4)
        XQW_batch = XQW_batch.reshape(B, L, self.width)
        return XQW_batch, batch_params_dict


class TTTMLPTritonSplit(TTTBase):
    def __init__(self, config: 'ModelArgs'):
        super().__init__(config)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, 4 * self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, 4 * self.head_dim))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 4 * self.head_dim, self.head_dim)))
        self.b2 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

    def init_weights(self, init_std: float):
        if self.config.fix_normal_initializer_range:
            for linear in (self.wq, self.wk, self.wv):
                nn.init.normal_(linear.weight, mean=0.0, std=self.config.initializer_range)
            nn.init.normal_(self.wo.weight, mean=0.0, std=self.config.initializer_range)
        else:
            for linear in (self.wq, self.wk, self.wv):
                nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
            nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

        # @xinhao: must explicitly initialize, otherwise become 0 after transferring from meta device to real device
        self.token_idx.copy_(1.0 / torch.arange(1, self.mini_batch_size + 1))
        self.post_norm.reset_parameters()
        self.ttt_norm_weight.data.copy_(torch.ones_like(self.ttt_norm_weight.data))
        self.ttt_norm_bias.data.copy_(torch.zeros_like(self.ttt_norm_bias.data))
        self.learnable_ttt_lr_weight.data.copy_(torch.randn_like(self.learnable_ttt_lr_weight.data) * 0.02)
        self.learnable_ttt_lr_bias.data.copy_(torch.zeros_like(self.learnable_ttt_lr_bias.data))
        self.W1.data.copy_(torch.randn_like(self.W1.data) * 0.02)
        self.b1.data.copy_(torch.zeros_like(self.b1.data))
        self.W2.data.copy_(torch.randn_like(self.W2.data) * 0.02)
        self.b2.data.copy_(torch.zeros_like(self.b2.data))

    def ttt(self, inputs, use_dual_form=True):
        mini_batch_size = self.mini_batch_size

        B = inputs["XV"].shape[0]
        num_mini_batch = inputs["XV"].shape[2]
        L = inputs["XV"].shape[2] * inputs["XV"].shape[3]
        device = inputs["XV"].device
        dtype = inputs["XV"].dtype

        W1_states = torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1))
        b1_states = torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1))
        W2_states = torch.tile(self.W2.unsqueeze(0), dims=(B, 1, 1, 1))
        b2_states = torch.tile(self.b2.unsqueeze(0), dims=(B, 1, 1, 1))

        checkpoint_group_size = (
            self.config.scan_checkpoint_group_size if self.config.scan_checkpoint_group_size > 0 else num_mini_batch
        )

        W1_last, b1_last, W2_last, b2_last, XQW_batch = TritonMLPSplit.apply(
            self.ttt_norm_weight,
            self.ttt_norm_bias,
            W1_states,
            b1_states,
            W2_states,
            b2_states,
            inputs["XQ"],
            inputs["XV"],
            inputs["XK"],
            inputs["eta"],
            checkpoint_group_size,
        )

        batch_params_dict = {
            "W1_states": W1_last,
            "b1_states": b1_last,
            "W2_states": W2_last,
            "b2_states": b2_last,
        }

        XQW_batch = XQW_batch.permute(0, 2, 3, 1, 4)
        XQW_batch = XQW_batch.reshape(B, L, self.width)
        return XQW_batch, batch_params_dict
    
