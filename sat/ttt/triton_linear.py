import triton
import triton.language as tl
import torch

from functools import partial
from torch.distributed._tensor import Partial, Replicate, Shard
from torch.distributed._tensor.experimental import local_map

from ttt.linear_forward import ttt_linear_scan_forward
from ttt.linear_backward import ttt_linear_scan_backward


class TritonLinear(torch.autograd.Function):
    @staticmethod
    @partial(
        local_map,
        out_placements=None,
        in_placements=None,
    )
    def forward(
        ctx,
        ttt_norm_weight,
        ttt_norm_bias,
        W1_init,
        b1_init,
        XQ_batch,
        XV_batch,
        XK_batch,
        eta_batch,
        checkpoint_group_size,
    ):
        B, NH, NC, CS, F = XQ_batch.shape
        K = NC // checkpoint_group_size

        device = XQ_batch.device
        comp_type = XQ_batch.dtype  # NOTE: For mixed precision, this is bfloat16
        accum_dtype = torch.float32

        # Output pointers
        W1_last = torch.empty(B, NH, F, F, device=device, dtype=accum_dtype)
        b1_last = torch.empty(B, NH, 1, F, device=device, dtype=accum_dtype)
        XQW_batch = torch.empty(B, NH, NC, CS, F, device=device, dtype=comp_type)

        # Context pointers
        W1_checkpoints = torch.empty(B, NH, K, F, F, device=device, dtype=accum_dtype)
        b1_checkpoints = torch.empty(B, NH, K, 1, F, device=device, dtype=accum_dtype)

        # Strides
        CS_F_stride = CS * F
        F_F_stride = F * F
        CS_CS_stride = CS * CS
        F_stride = F

        grid = (B, NH)

        ttt_linear_scan_forward[grid](
            # Scan inputs
            ttt_norm_weight.to(comp_type).contiguous(),
            ttt_norm_bias.to(comp_type).contiguous(),
            W1_init.to(torch.float32).contiguous(),
            b1_init.to(torch.float32).contiguous(),
            XQ_batch.contiguous(),
            XV_batch.contiguous(),
            XK_batch.contiguous(),
            eta_batch.contiguous(),
            # Outputs
            W1_last.contiguous(),
            b1_last.contiguous(),
            XQW_batch.contiguous(),
            # Context pointers
            W1_checkpoints.contiguous(),
            b1_checkpoints.contiguous(),
            # Strides
            CS_F_stride,
            F_F_stride,
            CS_CS_stride,
            F_stride,
            # Constant expressions
            NH,
            NC,
            CS,
            F,
            K,
            checkpoint_group_size,
        )

        ctx.save_for_backward(
            XQ_batch,
            XV_batch,
            XK_batch,
            eta_batch,
            ttt_norm_weight,
            ttt_norm_bias,
            W1_checkpoints,
            b1_checkpoints,
        )

        return W1_last, b1_last, XQW_batch

    @staticmethod
    @partial(
        local_map,
        out_placements=None,
        in_placements=None,
    )
    def backward(ctx, grad_L_W1_last, grad_L_b1_last, grad_L_XQW_batch):
        (
            XQ_batch,
            XV_batch,
            XK_batch,
            eta_batch,
            ttt_norm_weight,
            ttt_norm_bias,
            W1_checkpoints,
            b1_checkpoints,
        ) = ctx.saved_tensors

        B, NH, NC, CS, F = XQ_batch.shape
        K = W1_checkpoints.shape[2]
        checkpoint_group_size = NC // K

        device = XQ_batch.device
        comp_type = XQ_batch.dtype  # NOTE: For mixed precision, this is bfloat16
        accum_dtype = torch.float32

        # Intermediate buffers for each checkpoint group
        W1_init_group = torch.empty(B, NH, checkpoint_group_size, F, F, device=device, dtype=accum_dtype)
        b1_init_group = torch.empty(B, NH, checkpoint_group_size, 1, F, device=device, dtype=accum_dtype)
        x_hat_ln_group = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=comp_type)
        std_ln_group = torch.empty(B, NH, checkpoint_group_size, CS, 1, device=device, dtype=comp_type)
        grad_l_wrt_Z1_group = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=comp_type)
        Attn1_group = torch.empty(B, NH, checkpoint_group_size, CS, CS, device=device, dtype=comp_type)
        x_hat_fused_group = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=comp_type)
        grad_x_hat_fused_group = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=comp_type)
        grad_output_fused_group = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=comp_type)
        std_fused_group = torch.empty(B, NH, checkpoint_group_size, CS, 1, device=device, dtype=comp_type)
        XQW_mini_batch_group = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=comp_type)

        # NOTE: Sum over batch post-kernel to avoid sync barrier
        grad_L_ttt_norm_weight = torch.empty(B, NH, 1, F, device=device, dtype=accum_dtype)
        grad_L_ttt_norm_bias = torch.empty(B, NH, 1, F, device=device, dtype=accum_dtype)

        grad_L_W1_init = torch.empty(B, NH, F, F, device=device, dtype=accum_dtype)
        grad_L_b1_init = torch.empty(B, NH, 1, F, device=device, dtype=accum_dtype)

        grad_L_XQ = torch.empty(B, NH, NC, CS, F, device=device, dtype=comp_type)
        grad_L_XV = torch.empty(B, NH, NC, CS, F, device=device, dtype=comp_type)
        grad_L_XK = torch.empty(B, NH, NC, CS, F, device=device, dtype=comp_type)
        grad_L_eta = torch.empty(B, NH, NC, CS, CS, device=device, dtype=comp_type)

        CS_F_stride = CS * F
        F_F_stride = F * F
        CS_CS_stride = CS * CS
        F_stride = F

        grid = (B, NH)

        ttt_linear_scan_backward[grid](
            XQ_batch.contiguous(),
            XV_batch.contiguous(),
            XK_batch.contiguous(),
            eta_batch.contiguous(),
            ttt_norm_weight.contiguous(),
            ttt_norm_bias.contiguous(),
            W1_checkpoints.contiguous(),
            b1_checkpoints.contiguous(),
            # Upstream gradients
            grad_L_W1_last.to(torch.float32).contiguous(),
            grad_L_b1_last.to(torch.float32).contiguous(),
            grad_L_XQW_batch.contiguous(),
            # Intermediate buffers,
            XQW_mini_batch_group.contiguous(),
            W1_init_group.contiguous(),
            b1_init_group.contiguous(),
            x_hat_ln_group.contiguous(),
            std_ln_group.contiguous(),
            grad_l_wrt_Z1_group.contiguous(),
            Attn1_group.contiguous(),
            x_hat_fused_group.contiguous(),
            grad_x_hat_fused_group.contiguous(),
            grad_output_fused_group.contiguous(),
            std_fused_group.contiguous(),
            # Output buffers
            grad_L_ttt_norm_weight.contiguous(),
            grad_L_ttt_norm_bias.contiguous(),
            grad_L_W1_init.contiguous(),
            grad_L_b1_init.contiguous(),
            grad_L_XQ.contiguous(),
            grad_L_XV.contiguous(),
            grad_L_XK.contiguous(),
            grad_L_eta.contiguous(),
            # Strides
            CS_F_stride,
            F_F_stride,
            CS_CS_stride,
            F_stride,
            # Constant expressions
            NH,
            NC,
            CS,
            F,
            K,
            checkpoint_group_size,
        )

        grad_L_ttt_norm_weight = grad_L_ttt_norm_weight.sum(dim=0).squeeze(1)
        grad_L_ttt_norm_bias = grad_L_ttt_norm_bias.sum(dim=0).squeeze(1)

        return (
            grad_L_ttt_norm_weight.to(comp_type),
            grad_L_ttt_norm_bias.to(comp_type),
            grad_L_W1_init.to(comp_type),
            grad_L_b1_init.to(comp_type),
            grad_L_XQ.to(comp_type),
            grad_L_XV.to(comp_type),
            grad_L_XK.to(comp_type),
            grad_L_eta.to(comp_type),
            None,
        )
