import triton
import triton.language as tl
import torch

from functools import partial
from torch.distributed._tensor import Partial, Replicate, Shard
from torch.distributed._tensor.experimental import local_map

from ttt.mlp_forward_split import (
    ttt_mlp_stage_1 as fwd_stage_1,
    ttt_mlp_stage_2 as fwd_stage_2,
    ttt_mlp_stage_3 as fwd_stage_3,
)
from ttt.mlp_backward_split import (
    ttt_mlp_stage_1,
    ttt_mlp_stage_2,
    ttt_mlp_stage_3,
    ttt_mlp_backward_stage_1,
    ttt_mlp_backward_stage_2,
    ttt_mlp_backward_stage_3,
    ttt_mlp_backward_stage_4,
    ttt_mlp_backward_stage_5,
    ttt_mlp_backward_stage_6,
    ttt_mlp_backward_stage_7,
    ttt_mlp_backward_stage_8,
)


class TritonMLPSplit(torch.autograd.Function):
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
        W2_init,
        b2_init,
        XQ_batch,
        XV_batch,
        XK_batch,
        eta_batch,
        checkpoint_group_size,
    ):
        B, NH, NC, CS, F = XQ_batch.shape
        K = NC // checkpoint_group_size

        device = XQ_batch.device
        comp_dtype = XQ_batch.dtype  # NOTE: For mixed precision, this is bfloat16
        accum_dtype = torch.float32

        # Output pointers
        Z2_bar_ln = torch.empty(B, NH, NC, CS, F, device=device, dtype=comp_dtype).contiguous()

        # Context pointers
        W1_checkpoints = torch.empty(B, NH, K, F, F * 4, device=device, dtype=accum_dtype).contiguous()
        b1_checkpoints = torch.empty(B, NH, K, 1, F * 4, device=device, dtype=accum_dtype).contiguous()
        W2_checkpoints = torch.empty(B, NH, K, F * 4, F, device=device, dtype=accum_dtype).contiguous()
        b2_checkpoints = torch.empty(B, NH, K, 1, F, device=device, dtype=accum_dtype).contiguous()

        # Intermediates between kernels
        CS_F4_buffer_1 = torch.empty(
            B, NH, CS, F * 4, device=device, dtype=comp_dtype
        ).contiguous()  # Used for grad_l_wrt_Z1 / X2_bar
        CS_F_buffer_1 = torch.empty(
            B, NH, CS, F, device=device, dtype=comp_dtype
        ).contiguous()  # Used for grad_l_wrt_Z2
        CS_F4_buffer_2 = torch.empty(B, NH, CS, F * 4, device=device, dtype=comp_dtype).contiguous()  # Used for X2

        # Strides
        CS_F_stride = CS * F
        F_F4_stride = F * F * 4
        CS_CS_stride = CS * CS
        F_stride = F
        F4_stride = F * 4

        grid = (B, NH)

        # Cast and make inputs contiguous
        XQ_batch = XQ_batch.contiguous()
        XV_batch = XV_batch.contiguous()
        XK_batch = XK_batch.contiguous()
        eta_batch = eta_batch.to(comp_dtype).contiguous()

        W1_init = W1_init.to(torch.float32).contiguous()
        b1_init = b1_init.to(torch.float32).contiguous()
        W2_init = W2_init.to(torch.float32).contiguous()
        b2_init = b2_init.to(torch.float32).contiguous()

        for i in range(NC):

            # Save checkpoints
            if i % checkpoint_group_size == 0:
                W1_checkpoints[:, :, i // checkpoint_group_size] = W1_init
                b1_checkpoints[:, :, i // checkpoint_group_size] = b1_init
                W2_checkpoints[:, :, i // checkpoint_group_size] = W2_init
                b2_checkpoints[:, :, i // checkpoint_group_size] = b2_init

            fwd_stage_1[grid](
                # Scan inputs
                ttt_norm_weight,
                ttt_norm_bias,
                W1_init,
                b1_init,
                W2_init,
                b2_init,
                XV_batch,
                XK_batch,
                # Outputs
                CS_F4_buffer_1,
                CS_F_buffer_1,
                CS_F4_buffer_2,
                # Strides
                CS_F_stride,
                F_F4_stride,
                F_stride,
                F4_stride,
                # Constant expressions
                NH,
                NC,
                CS,
                F,
                # Index
                i,
                num_warps=8,
                num_stages=4,
                num_ctas=1,
            )

            fwd_stage_2[grid](
                # Scan inputs
                W1_init,
                b1_init,
                XQ_batch,
                XK_batch,
                eta_batch,
                # Intermediates
                CS_F4_buffer_1,
                # Strides
                CS_F_stride,
                F_F4_stride,
                CS_CS_stride,
                F_stride,
                F4_stride,
                # Constant expressions
                NH,
                NC,
                CS,
                F,
                # Index
                i,
                num_warps=8,
                num_stages=4,
                num_ctas=1,
            )

            fwd_stage_3[grid](
                # Scan inputs
                ttt_norm_weight,
                ttt_norm_bias,
                W2_init,
                b2_init,
                eta_batch,
                # Intermediates
                CS_F_buffer_1,
                CS_F4_buffer_2,
                CS_F4_buffer_1,
                # Outputs
                Z2_bar_ln,
                # Strides
                CS_F_stride,
                F_F4_stride,
                CS_CS_stride,
                F_stride,
                F4_stride,
                # Constant expressions
                NH,
                NC,
                CS,
                F,
                # Index
                i,
                num_warps=8,
                num_stages=4,
                num_ctas=1,
            )

        XQW_batch = Z2_bar_ln + XQ_batch

        ctx.save_for_backward(
            XQ_batch,
            XV_batch,
            XK_batch,
            eta_batch,
            ttt_norm_weight,
            ttt_norm_bias,
            W1_checkpoints,
            b1_checkpoints,
            W2_checkpoints,
            b2_checkpoints,
        )

        return (
            W1_init.to(comp_dtype),
            b1_init.to(comp_dtype),
            W2_init.to(comp_dtype),
            b2_init.to(comp_dtype),
            XQW_batch.to(comp_dtype),
        )

    @staticmethod
    @partial(
        local_map,
        out_placements=None,
        in_placements=None,
    )
    def backward(ctx, grad_L_W1_last, grad_L_b1_last, grad_L_W2_last, grad_L_b2_last, grad_L_XQW_batch):
        (
            XQ_batch,
            XV_batch,
            XK_batch,
            eta_batch,
            ttt_norm_weight,
            ttt_norm_bias,
            W1_checkpoints,
            b1_checkpoints,
            W2_checkpoints,
            b2_checkpoints,
        ) = ctx.saved_tensors

        B, NH, NC, CS, F = XQ_batch.shape
        K = W1_checkpoints.shape[2]
        checkpoint_group_size = NC // K

        device = XQ_batch.device
        comp_dtype = XQ_batch.dtype  # NOTE: For mixed precision, this is bfloat16
        accum_dtype = torch.float32

        W1_checkpoints = W1_checkpoints.permute(2, 0, 1, 3, 4).contiguous()
        b1_checkpoints = b1_checkpoints.permute(2, 0, 1, 3, 4).contiguous()
        W2_checkpoints = W2_checkpoints.permute(2, 0, 1, 3, 4).contiguous()
        b2_checkpoints = b2_checkpoints.permute(2, 0, 1, 3, 4).contiguous()

        # Cast upstream grads
        grad_L_W1_last = grad_L_W1_last.to(accum_dtype).contiguous()
        grad_L_b1_last = grad_L_b1_last.to(accum_dtype).contiguous()
        grad_L_W2_last = grad_L_W2_last.to(accum_dtype).contiguous()
        grad_L_b2_last = grad_L_b2_last.to(accum_dtype).contiguous()
        grad_L_XQW_batch = grad_L_XQW_batch.to(accum_dtype).contiguous()

        # Intermediate buffers
        W1_init_group = torch.empty(B, NH, checkpoint_group_size, F, F * 4, device=device, dtype=accum_dtype)
        b1_init_group = torch.empty(B, NH, checkpoint_group_size, 1, F * 4, device=device, dtype=accum_dtype)
        W2_init_group = torch.empty(B, NH, checkpoint_group_size, F * 4, F, device=device, dtype=accum_dtype)
        b2_init_group = torch.empty(B, NH, checkpoint_group_size, 1, F, device=device, dtype=accum_dtype)

        x_hat_ln_group = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=comp_dtype)
        std_ln_group = torch.empty(B, NH, checkpoint_group_size, CS, 1, device=device, dtype=comp_dtype)
        Attn1_group = torch.empty(B, NH, checkpoint_group_size, CS, CS, device=device, dtype=comp_dtype)
        Attn2_group = torch.empty(B, NH, checkpoint_group_size, CS, CS, device=device, dtype=comp_dtype)

        X2_group = torch.empty(B, NH, checkpoint_group_size, CS, F * 4, device=device, dtype=comp_dtype)
        Z1_group = torch.empty(B, NH, checkpoint_group_size, CS, F * 4, device=device, dtype=comp_dtype)
        Z1_bar_group = torch.empty(B, NH, checkpoint_group_size, CS, F * 4, device=device, dtype=comp_dtype)
        X2_bar_group = torch.empty(B, NH, checkpoint_group_size, CS, F * 4, device=device, dtype=comp_dtype)

        grad_l_wrt_Z2_group = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=comp_dtype)
        grad_l_wrt_Z1_group = torch.empty(B, NH, checkpoint_group_size, CS, F * 4, device=device, dtype=comp_dtype)
        x_hat_fused_group = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=comp_dtype)
        grad_x_hat_fused_group = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=comp_dtype)
        grad_output_fused_group = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=comp_dtype)
        std_fused_group = torch.empty(B, NH, checkpoint_group_size, CS, 1, device=device, dtype=comp_dtype)

        # Intermediate buffers between stages
        grad_L_grad_l_wrt_Z2 = torch.empty(B, NH, CS, F, device=device, dtype=comp_dtype)
        grad_L_eta_Attn2 = torch.empty(B, NH, CS, CS, device=device, dtype=comp_dtype)
        grad_L_XK_mini_batch = torch.zeros(B, NH, CS, F, device=device, dtype=comp_dtype)
        grad_L_Z1_bar = torch.empty(B, NH, CS, F * 4, device=device, dtype=comp_dtype)
        grad_L_Z1 = torch.empty(B, NH, CS, F * 4, device=device, dtype=comp_dtype)
        grad_L_Z2 = torch.empty(B, NH, CS, F, device=device, dtype=comp_dtype)
        grad_L_Z2_bar = torch.empty(B, NH, CS, F, device=device, dtype=comp_dtype)
        grad_l_wrt_Z1_Last = torch.empty(B, NH, F, CS, device=device, dtype=comp_dtype)
        grad_L_grad_l_wrt_Z1 = torch.empty(B, NH, CS, F * 4, device=device, dtype=comp_dtype)
        grad_L_W2_init = torch.empty(B, NH, F * 4, F, device=device, dtype=accum_dtype)
        grad_L_b1_init = torch.empty(B, NH, checkpoint_group_size, 1, F * 4, device=device, dtype=accum_dtype)

        # Final gradients
        grad_L_XQ = torch.empty(B, NH, NC, CS, F, device=device, dtype=accum_dtype)
        grad_L_XV = torch.empty(B, NH, NC, CS, F, device=device, dtype=accum_dtype)
        grad_L_XK = torch.empty(B, NH, NC, CS, F, device=device, dtype=accum_dtype)
        grad_L_eta = torch.zeros(B, NH, NC, CS, CS, device=device, dtype=accum_dtype)

        # NOTE: Sum over batch post-kernel to avoid sync barrier
        grad_L_ttt_norm_weight = torch.zeros(B, NH, 1, F, device=device, dtype=accum_dtype)
        grad_L_ttt_norm_bias = torch.zeros(B, NH, 1, F, device=device, dtype=accum_dtype)

        CS_stride = CS
        CS_F_stride = CS * F
        CS_CS_stride = CS * CS
        F_stride = F
        F4_stride = F * 4
        F_F4_stride = F * F * 4

        grid = (B, NH)

        for checkpoint_idx in range(K - 1, -1, -1):
            W1_init = W1_checkpoints[checkpoint_idx, :, :, :, :].contiguous()
            b1_init = b1_checkpoints[checkpoint_idx, :, :, :, :].contiguous()
            W2_init = W2_checkpoints[checkpoint_idx, :, :, :, :].contiguous()
            b2_init = b2_checkpoints[checkpoint_idx, :, :, :, :].contiguous()

            # Recover forward activations for current checkpoint group
            for mini_batch_idx_in_group in range(checkpoint_group_size):
                mini_batch_idx = checkpoint_idx * checkpoint_group_size + mini_batch_idx_in_group

                ttt_mlp_stage_1[grid](
                    # Scan inputs
                    ttt_norm_weight,
                    ttt_norm_bias,
                    W1_init,
                    b1_init,
                    W2_init,
                    b2_init,
                    XV_batch,
                    XK_batch,
                    # Intermediate buffers
                    W1_init_group,
                    b1_init_group,
                    W2_init_group,
                    b2_init_group,
                    x_hat_ln_group,
                    std_ln_group,
                    Attn1_group,
                    Attn2_group,
                    X2_group,
                    Z1_group,
                    Z1_bar_group,
                    X2_bar_group,
                    grad_l_wrt_Z2_group,
                    grad_l_wrt_Z1_group,
                    x_hat_fused_group,
                    grad_x_hat_fused_group,
                    grad_output_fused_group,
                    std_fused_group,
                    # Strides
                    CS_F_stride,
                    F_F4_stride,
                    F_stride,
                    F4_stride,
                    # Constant expressions
                    NH,
                    NC,
                    CS,
                    F,
                    checkpoint_group_size,
                    # Index
                    mini_batch_idx,
                    mini_batch_idx_in_group,
                    num_warps=8,
                    num_stages=4,
                    num_ctas=1,
                )

                ttt_mlp_stage_2[grid](
                    # Scan inputs
                    W1_init,
                    b1_init,
                    XQ_batch,
                    XK_batch,
                    eta_batch,
                    # Intermediate buffers
                    W1_init_group,
                    b1_init_group,
                    W2_init_group,
                    b2_init_group,
                    x_hat_ln_group,
                    std_ln_group,
                    Attn1_group,
                    Attn2_group,
                    X2_group,
                    Z1_group,
                    Z1_bar_group,
                    X2_bar_group,
                    grad_l_wrt_Z2_group,
                    grad_l_wrt_Z1_group,
                    x_hat_fused_group,
                    grad_x_hat_fused_group,
                    grad_output_fused_group,
                    std_fused_group,
                    # Strides
                    CS_F_stride,
                    F_F4_stride,
                    CS_CS_stride,
                    F_stride,
                    F4_stride,
                    # Constant expressions
                    NH,
                    NC,
                    CS,
                    F,
                    checkpoint_group_size,
                    # Index
                    mini_batch_idx,
                    mini_batch_idx_in_group,
                    num_warps=8,
                    num_stages=4,
                    num_ctas=1,
                )

                ttt_mlp_stage_3[grid](
                    # Scan inputs
                    ttt_norm_weight,
                    ttt_norm_bias,
                    W2_init,
                    b2_init,
                    eta_batch,
                    # Intermediate buffers
                    W1_init_group,
                    b1_init_group,
                    W2_init_group,
                    b2_init_group,
                    x_hat_ln_group,
                    std_ln_group,
                    Attn1_group,
                    Attn2_group,
                    X2_group,
                    Z1_group,
                    Z1_bar_group,
                    X2_bar_group,
                    grad_l_wrt_Z2_group,
                    grad_l_wrt_Z1_group,
                    x_hat_fused_group,
                    grad_x_hat_fused_group,
                    grad_output_fused_group,
                    std_fused_group,
                    # Strides
                    CS_F_stride,
                    F_F4_stride,
                    CS_CS_stride,
                    F_stride,
                    F4_stride,
                    # Constant expressions
                    NH,
                    NC,
                    CS,
                    F,
                    checkpoint_group_size,
                    # Index
                    mini_batch_idx,
                    mini_batch_idx_in_group,
                    num_warps=8,
                    num_stages=4,
                    num_ctas=1,
                )

            # Run backward pass for current checkpoint group
            for mini_batch_in_group_idx in range(checkpoint_group_size - 1, -1, -1):

                ttt_mlp_backward_stage_1[grid](
                    ttt_norm_weight,
                    # Upstream gradients
                    grad_L_XQW_batch,
                    grad_L_W1_last,
                    # Intermediate buffers
                    x_hat_ln_group,
                    std_ln_group,
                    grad_l_wrt_Z1_group,
                    # Other stages
                    grad_L_Z2_bar,
                    grad_l_wrt_Z1_Last,
                    # Output buffers
                    grad_L_ttt_norm_weight,
                    grad_L_ttt_norm_bias,
                    # Strides
                    CS_F_stride,
                    F_F4_stride,
                    F_stride,
                    F4_stride,
                    # Constants
                    NH,
                    NC,
                    CS,
                    F,
                    checkpoint_group_size,
                    checkpoint_idx,
                    mini_batch_in_group_idx,
                    num_warps=8,
                )

                ttt_mlp_backward_stage_2[grid](
                    XK_batch,
                    eta_batch,
                    # Upstream gradients
                    grad_L_W1_last,
                    grad_L_b1_last,
                    # Intermediate buffers
                    W2_init_group,
                    Attn1_group,
                    Attn2_group,
                    X2_group,
                    Z1_bar_group,
                    grad_l_wrt_Z2_group,
                    # Other stages
                    grad_L_grad_l_wrt_Z1,
                    grad_L_eta_Attn2,
                    grad_L_Z1_bar,
                    grad_L_Z2_bar,
                    # Output buffers
                    grad_L_eta,
                    # Strides
                    CS_F_stride,
                    F_F4_stride,
                    CS_CS_stride,
                    F4_stride,
                    # Constants
                    NH,
                    NC,
                    CS,
                    F,
                    checkpoint_group_size,
                    checkpoint_idx,
                    mini_batch_in_group_idx,
                    num_warps=8,
                )

                ttt_mlp_backward_stage_3[grid](
                    XQ_batch,
                    eta_batch,
                    # Upstream gradients
                    grad_L_W1_last,
                    grad_L_b1_last,
                    # Intermediate buffers
                    grad_l_wrt_Z1_group,
                    # Other stages
                    grad_L_XK_mini_batch,
                    grad_L_Z1_bar,
                    grad_l_wrt_Z1_Last,
                    grad_L_b1_init,
                    # Strides
                    CS_F_stride,
                    F_F4_stride,
                    CS_CS_stride,
                    F4_stride,
                    # Constants
                    NH,
                    NC,
                    CS,
                    F,
                    checkpoint_group_size,
                    checkpoint_idx,
                    mini_batch_in_group_idx,
                    num_warps=8,
                )

                ttt_mlp_backward_stage_4[grid](
                    eta_batch,
                    # Upstream gradients
                    grad_L_W2_last,
                    grad_L_b2_last,
                    # Intermediate buffers
                    W2_init_group,
                    Attn2_group,
                    X2_group,
                    Z1_group,
                    grad_l_wrt_Z2_group,
                    # Other stages
                    grad_L_grad_l_wrt_Z2,
                    grad_L_Z1,
                    grad_L_Z2_bar,
                    grad_L_grad_l_wrt_Z1,
                    # Strides
                    CS_F_stride,
                    F_F4_stride,
                    CS_CS_stride,
                    F_stride,
                    # Constants
                    NH,
                    NC,
                    CS,
                    F,
                    checkpoint_group_size,
                    checkpoint_idx,
                    mini_batch_in_group_idx,
                    num_warps=8,
                )

                ttt_mlp_backward_stage_5[grid](
                    XK_batch,
                    eta_batch,
                    # Upstream gradients
                    grad_L_b1_last,
                    grad_L_W2_last,
                    grad_L_b2_last,
                    grad_L_XQW_batch,
                    # Intermediate buffers
                    W1_init_group,
                    Attn1_group,
                    X2_group,
                    Z1_group,
                    X2_bar_group,
                    grad_l_wrt_Z2_group,
                    grad_l_wrt_Z1_group,
                    # Other stages
                    grad_L_Z1_bar,
                    grad_L_Z2_bar,
                    grad_l_wrt_Z1_Last,
                    grad_L_grad_l_wrt_Z1,
                    grad_L_W2_init,
                    # Output buffers
                    grad_L_XQ,
                    grad_L_eta,
                    # Strides
                    CS_F_stride,
                    F_F4_stride,
                    CS_CS_stride,
                    F_stride,
                    F4_stride,
                    # Constants
                    NH,
                    NC,
                    CS,
                    F,
                    checkpoint_group_size,
                    checkpoint_idx,
                    mini_batch_in_group_idx,
                    num_warps=8,
                )

                ttt_mlp_backward_stage_6[grid](
                    ttt_norm_weight,
                    # Upstream gradients
                    grad_L_b2_last,
                    # Intermediate buffers
                    X2_group,
                    grad_l_wrt_Z2_group,
                    x_hat_fused_group,
                    grad_x_hat_fused_group,
                    grad_output_fused_group,
                    std_fused_group,
                    # Other stages
                    grad_L_grad_l_wrt_Z2,
                    grad_L_XK_mini_batch,
                    grad_L_Z2,
                    grad_L_W2_init,
                    # Output buffers
                    grad_L_ttt_norm_weight,
                    grad_L_ttt_norm_bias,
                    grad_L_XV,
                    grad_L_XK,
                    # Strides
                    CS_F_stride,
                    F_F4_stride,
                    F_stride,
                    # Constants
                    NH,
                    NC,
                    CS,
                    F,
                    checkpoint_group_size,
                    checkpoint_idx,
                    mini_batch_in_group_idx,
                    num_warps=8,
                )

                ttt_mlp_backward_stage_7[grid](
                    eta_batch,
                    # Upstream gradients
                    grad_L_W2_last,
                    # Intermediate buffers
                    W2_init_group,
                    Z1_group,
                    X2_bar_group,
                    grad_l_wrt_Z2_group,
                    # Other stages
                    grad_L_eta_Attn2,
                    grad_L_Z1,
                    grad_L_Z2,
                    # Strides
                    CS_F_stride,
                    F_F4_stride,
                    CS_CS_stride,
                    # Constants
                    NH,
                    NC,
                    CS,
                    F,
                    checkpoint_group_size,
                    checkpoint_idx,
                    mini_batch_in_group_idx,
                    num_warps=8,
                )

                ttt_mlp_backward_stage_8[grid](
                    XK_batch,
                    # Upstream gradients
                    grad_L_W1_last,
                    grad_L_b1_last,
                    # Intermediate buffers
                    W1_init_group,
                    # Other stages
                    grad_L_Z1,
                    grad_L_b1_init,
                    # Output buffers
                    grad_L_XK,
                    # Strides
                    CS_F_stride,
                    F_F4_stride,
                    F4_stride,
                    # Constants
                    NH,
                    NC,
                    CS,
                    F,
                    checkpoint_group_size,
                    checkpoint_idx,
                    mini_batch_in_group_idx,
                    num_warps=8,
                )

                # @Daniel: Hack around extra buffer requirement
                temp = grad_L_W2_last
                grad_L_W2_last = grad_L_W2_init
                grad_L_W2_init = temp

        grad_L_ttt_norm_weight = grad_L_ttt_norm_weight.sum(dim=0).squeeze(1)
        grad_L_ttt_norm_bias = grad_L_ttt_norm_bias.sum(dim=0).squeeze(1)

        return (
            grad_L_ttt_norm_weight.to(comp_dtype),
            grad_L_ttt_norm_bias.to(comp_dtype),
            grad_L_W1_last.to(comp_dtype),
            grad_L_b1_last.to(comp_dtype),
            grad_L_W2_last.to(comp_dtype),
            grad_L_b2_last.to(comp_dtype),
            grad_L_XQ.to(comp_dtype),
            grad_L_XV.to(comp_dtype),
            grad_L_XK.to(comp_dtype),
            grad_L_eta.to(comp_dtype),
            None,
        )
