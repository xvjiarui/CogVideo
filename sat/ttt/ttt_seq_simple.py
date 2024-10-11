from functools import partial
import copy

import torch
import torch.nn as nn

from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from ttt.ttt_layer import ModelArgs, TTTLinear, TTTMLP, TTTLinearTriton, TTTMLPTritonSplit

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

class TTTWrapper(nn.Module):
    def __init__(self, d_model, ssm_cfg):
        super().__init__()
        ssm_layer = ssm_cfg["layer"]
        # TODO(xvjiarui): add make them not causal
        # if ssm_layer == "TTTLinear":
        #     mixer_cls = TTTLinear
        # elif ssm_layer == "TTTMLP":
        #     mixer_cls = TTTMLP
        if ssm_layer == "TTTLinearTriton":
            mixer_cls = TTTLinearTriton
        elif ssm_layer == "TTTMLPTritonSplit":
            mixer_cls = TTTMLPTritonSplit
        else:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support TTTLinear, TTTMLP, TTTLinearTriton and TTTMLPTritonSplit")
        config = ModelArgs(
            dim=d_model,
            n_heads=d_model // ssm_cfg['headdim'],
            # TODO(xvjiarui): make it dynamic
            max_seq_len=64_000,
            ttt_base_lr=ssm_cfg.get("ttt_base_lr", 1.0),
            mini_batch_size=ssm_cfg.get("mini_batch_size", 16),
            scan_checkpoint_group_size=ssm_cfg.get("scan_checkpoint_group_size", 4),
        )
        self.model_args = config
        self.ttt = mixer_cls(config)
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)
    
    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(
            self.model_args.dim // self.model_args.n_heads,
            # Need to compute until at least the max token limit for generation
            # (use 2x max sequence length to be safe)
            self.model_args.max_seq_len * 2,
            self.model_args.rope_theta,
        )
    def forward(self, x, inference_params=None):
        assert inference_params is None, "Inference params not supported"
        return self.ttt(x, self.freqs_cis)
        

def create_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    # Create a copy of the config to modify
    ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
    mixer_cls = partial(
        TTTWrapper, ssm_cfg=ssm_cfg
    )
    factory_kwargs = {"device": device, "dtype": dtype}
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block
