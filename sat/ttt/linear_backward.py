import torch
import triton
import numpy as np
import triton.language as tl


@triton.jit
def ttt_linear_mini_batch_forward(
    W1_init,
    b1_init,
    ln_weight,
    ln_bias,
    XQ_mini_batch,
    XK_mini_batch,
    XV_mini_batch,
    eta_mini_batch,
    last_eta_mini_batch,
    CS: tl.constexpr,
    F: tl.constexpr,
    comp_dtype,
    accum_dtype,
):
    # Stage 1: MatMul
    Z1 = (tl.dot(XK_mini_batch, W1_init.to(comp_dtype)) + b1_init).to(comp_dtype)
    reconstruction_target = XV_mini_batch - XK_mini_batch

    # Stage 2: LnFusedL2BWD
    mu_fused = (tl.sum(Z1, axis=1) / F)[:, None].to(comp_dtype)
    var_fused = (tl.sum((Z1 - mu_fused) * (Z1 - mu_fused), axis=1) / F)[:, None].to(comp_dtype)

    std_fused = tl.sqrt(var_fused + 1e-6).to(comp_dtype)
    x_hat_fused = ((Z1 - mu_fused) / std_fused).to(comp_dtype)

    y = ln_weight * x_hat_fused + ln_bias
    grad_output_fused = y - reconstruction_target
    grad_x_hat_fused = grad_output_fused * ln_weight

    grad_l_wrt_Z1 = (
        (1.0 / F)
        * (
            F * grad_x_hat_fused
            - tl.sum(grad_x_hat_fused, axis=1)[:, None]
            - x_hat_fused * tl.sum(grad_x_hat_fused * x_hat_fused, axis=1)[:, None]
        )
        / std_fused
    ).to(comp_dtype)

    # Stage 3: Dual Form
    mask = tl.arange(0, CS)[:, None] >= tl.arange(0, CS)[None, :]
    Attn1 = tl.where(mask, tl.dot(XQ_mini_batch, tl.trans(XK_mini_batch)), 0).to(comp_dtype)
    b1_bar = (b1_init - tl.dot(tl.where(mask, eta_mini_batch, 0).to(comp_dtype), grad_l_wrt_Z1)).to(comp_dtype)
    Z1_bar = (
        tl.dot(XQ_mini_batch, W1_init.to(comp_dtype))
        - tl.dot((eta_mini_batch * Attn1).to(comp_dtype), grad_l_wrt_Z1)
        + b1_bar
    ).to(comp_dtype)

    # NOTE: Accumulation in FP32
    W1_last = W1_init - tl.dot(tl.trans(last_eta_mini_batch * XK_mini_batch).to(comp_dtype), grad_l_wrt_Z1)
    b1_last = b1_init - tl.sum(last_eta_mini_batch * grad_l_wrt_Z1, axis=0)[None, :]

    # Stage 4: LN
    mu_ln = (tl.sum(Z1_bar, axis=1)[:, None] / F).to(comp_dtype)
    var_ln = (tl.sum((Z1_bar - mu_ln) * (Z1_bar - mu_ln), axis=1)[:, None] / F).to(comp_dtype)
    std_ln = tl.sqrt(var_ln + 1e-6).to(comp_dtype)
    x_hat_ln = ((Z1_bar - mu_ln) / std_ln).to(comp_dtype)

    Z1_bar_ln = ln_weight * x_hat_ln + ln_bias

    XQW_mini_batch = XQ_mini_batch + Z1_bar_ln

    return (
        XQW_mini_batch,
        W1_last,
        b1_last,
        x_hat_ln,
        std_ln,
        grad_l_wrt_Z1,
        Attn1,
        x_hat_fused,
        grad_x_hat_fused,
        grad_output_fused,
        std_fused,
    )


@triton.jit
def ttt_linear_mini_batch_backward(
    # MatMul
    XQ_mini_batch,
    XK_mini_batch,
    W1_init,
    b1_init,
    # LnFusedL2BWD
    ln_weight,
    ln_bias,
    std_fused,
    x_hat_fused,
    grad_output_fused,
    grad_x_hat_fused,
    grad_l_wrt_Z1,
    # Dual Form
    Attn1,
    eta_mini_batch,
    last_eta_mini_batch,
    # LN
    std_ln,
    x_hat_ln,
    # Upstream gradients
    grad_L_W1_last,
    grad_L_b1_last,
    grad_L_XQW_mini_batch,
    # Strides
    CS_F_stride: tl.constexpr,
    F_F_stride: tl.constexpr,
    CS_CS_stride: tl.constexpr,
    F_stride: tl.constexpr,
    # Constant expressions
    NH: tl.constexpr,
    CS: tl.constexpr,
    F: tl.constexpr,
    comp_dtype,
    accum_dtype,
):
    mask = tl.arange(0, CS)[:, None] >= tl.arange(0, CS)[None, :]

    # Stage 4: LN
    grad_L_ln_weight_ln = tl.sum(grad_L_XQW_mini_batch * x_hat_ln, axis=0).to(comp_dtype)
    grad_L_ln_bias_ln = tl.sum(grad_L_XQW_mini_batch, axis=0).to(comp_dtype)

    grad_L_x_hat_ln = grad_L_XQW_mini_batch * ln_weight
    grad_L_Z1_bar = (
        (1.0 / F)
        * (
            F * grad_L_x_hat_ln
            - tl.sum(grad_L_x_hat_ln, axis=1)[:, None]
            - x_hat_ln * tl.sum(grad_L_x_hat_ln * x_hat_ln, axis=1)[:, None]
        )
        / std_ln
    ).to(comp_dtype)

    # Stage 3: Dual Form
    grad_L_grad_l_wrt_Z1 = (
        -(tl.dot(tl.trans(tl.where(mask, eta_mini_batch, 0).to(comp_dtype)), grad_L_Z1_bar))
        - (tl.dot(tl.trans(eta_mini_batch * Attn1).to(comp_dtype), grad_L_Z1_bar))
        - (tl.dot((last_eta_mini_batch * XK_mini_batch).to(comp_dtype), grad_L_W1_last.to(comp_dtype)))
        - (last_eta_mini_batch * grad_L_b1_last.to(comp_dtype))
    ).to(comp_dtype)

    grad_L_b1_init = grad_L_b1_last + tl.sum(grad_L_Z1_bar, axis=0)
    grad_L_W1_init = grad_L_W1_last + tl.trans(tl.dot(tl.trans(grad_L_Z1_bar), XQ_mini_batch))

    grad_L_eta_Attn1 = tl.dot(grad_L_Z1_bar, tl.trans(grad_l_wrt_Z1)).to(comp_dtype)

    grad_L_XQ_mini_batch = (
        -tl.dot(tl.where(mask, grad_L_eta_Attn1 * eta_mini_batch, 0).to(comp_dtype), XK_mini_batch)
        + tl.dot(grad_L_Z1_bar, tl.trans(W1_init.to(comp_dtype)))
    ).to(comp_dtype)

    grad_l_wrt_Z1_Last = tl.dot(grad_l_wrt_Z1, tl.trans(grad_L_W1_last.to(comp_dtype))).to(comp_dtype)

    grad_L_XK_mini_batch = (
        -tl.dot(tl.trans(tl.where(mask, grad_L_eta_Attn1 * eta_mini_batch, 0).to(comp_dtype)), XQ_mini_batch)
        - grad_l_wrt_Z1_Last * last_eta_mini_batch
    ).to(comp_dtype)

    grad_L_last_eta_in_mini_batch = tl.sum(
        -(grad_l_wrt_Z1_Last * XK_mini_batch) - (grad_L_b1_last.to(comp_dtype) * grad_l_wrt_Z1), axis=1
    )[None, :].to(comp_dtype)

    last_mini_batch_mask = tl.arange(0, CS)[:, None] == CS - 1
    grad_L_eta_mini_batch = -tl.where(mask, grad_L_eta_Attn1, 0).to(comp_dtype) - (Attn1 * grad_L_eta_Attn1)
    grad_L_eta_mini_batch += tl.where(last_mini_batch_mask, grad_L_last_eta_in_mini_batch, 0).to(comp_dtype)

    # Stage 2: LnFusedL2BWD
    grad_L_grad_x_hat_fused = (
        (1.0 / std_fused) * grad_L_grad_l_wrt_Z1
        + (1.0 / F) * tl.sum(-grad_L_grad_l_wrt_Z1 * (1.0 / std_fused), axis=1)[:, None]
        + (1.0 / F) * x_hat_fused * tl.sum(-grad_L_grad_l_wrt_Z1 * (1.0 / std_fused) * x_hat_fused, axis=1)[:, None]
    ).to(comp_dtype)

    grad_L_y = ln_weight * grad_L_grad_x_hat_fused

    grad_L_ln_weight_fused = tl.sum(grad_output_fused * grad_L_grad_x_hat_fused + grad_L_y * x_hat_fused, axis=0).to(
        comp_dtype
    )
    grad_L_ln_bias_fused = tl.sum(grad_L_y, axis=0).to(comp_dtype)

    grad_L_x_hat_fused = (
        grad_L_y * ln_weight
        + (1.0 / F)
        * grad_x_hat_fused
        * tl.sum(-grad_L_grad_l_wrt_Z1 * (1.0 / std_fused) * x_hat_fused, axis=1)[:, None]
        + (1.0 / F)
        * tl.sum(grad_x_hat_fused * x_hat_fused, axis=1)[:, None]
        * (-grad_L_grad_l_wrt_Z1 * (1.0 / std_fused))
    ).to(comp_dtype)

    grad_L_std = (
        -grad_L_x_hat_fused * (x_hat_fused / std_fused)
        - (grad_L_grad_l_wrt_Z1 * ((grad_l_wrt_Z1 * std_fused) / (std_fused * std_fused)))
    ).to(comp_dtype)

    grad_L_Z1 = (
        grad_L_x_hat_fused * (1.0 / std_fused)
        - (1.0 / F) * tl.sum(grad_L_x_hat_fused, axis=1)[:, None] * (1.0 / std_fused)
        + (1.0 / F) * tl.sum(grad_L_std, axis=1)[:, None] * x_hat_fused
    ).to(comp_dtype)

    grad_L_reconstruction_target = -ln_weight * grad_L_grad_x_hat_fused

    # Stage 1: MatMul
    grad_L_XQ = grad_L_XQW_mini_batch + grad_L_XQ_mini_batch
    grad_L_XV = grad_L_reconstruction_target

    grad_L_XK = (
        -grad_L_reconstruction_target + grad_L_XK_mini_batch + tl.dot(grad_L_Z1, tl.trans(W1_init.to(comp_dtype)))
    ).to(comp_dtype)

    grad_L_W1_init = grad_L_W1_init + tl.trans((tl.dot(tl.trans(grad_L_Z1), XK_mini_batch)))
    grad_L_b1_init = grad_L_b1_init + (tl.sum(grad_L_Z1, axis=0))

    # NOTE: Sum over batch post-kernel to avoid sync barrier
    grad_L_ttt_norm_weight = ((grad_L_ln_weight_ln + grad_L_ln_weight_fused)[None, :]).to(accum_dtype)
    grad_L_ttt_norm_bias = ((grad_L_ln_bias_ln + grad_L_ln_bias_fused)[None, :]).to(accum_dtype)

    return (
        grad_L_ttt_norm_weight,
        grad_L_ttt_norm_bias,
        grad_L_W1_init,
        grad_L_b1_init,
        grad_L_XQ,
        grad_L_XV,
        grad_L_XK,
        grad_L_eta_mini_batch,
    )


@triton.jit
def ttt_linear_scan_backward(
    XQ_batch_ptr,
    XV_batch_ptr,
    XK_batch_ptr,
    eta_batch_ptr,
    ttt_norm_weight_ptr,
    ttt_norm_bias_ptr,
    W1_checkpoints_ptr,
    b1_checkpoints_ptr,
    # Upstream gradients
    grad_L_W1_last_ptr,
    grad_L_b1_last_ptr,
    grad_L_XQW_mini_batch_ptr,
    # Intermediate buffers
    XQW_mini_batch_group_ptr,
    W1_init_group_ptr,
    b1_init_group_ptr,
    x_hat_ln_group_ptr,
    std_ln_group_ptr,
    grad_l_wrt_Z1_group_ptr,
    Attn1_group_ptr,
    x_hat_fused_group_ptr,
    grad_x_hat_fused_group_ptr,
    grad_output_fused_group_ptr,
    std_fused_group_ptr,
    # Output buffers
    grad_L_ttt_norm_weight_ptr,
    grad_L_ttt_norm_bias_ptr,
    grad_L_W1_init_ptr,
    grad_L_b1_init_ptr,
    grad_L_XQ_ptr,
    grad_L_XV_ptr,
    grad_L_XK_ptr,
    grad_L_eta_ptr,
    # Strides
    CS_F_stride: tl.constexpr,
    F_F_stride: tl.constexpr,
    CS_CS_stride: tl.constexpr,
    F_stride: tl.constexpr,
    # Constant expressions
    NH: tl.constexpr,
    NC: tl.constexpr,
    CS: tl.constexpr,
    F: tl.constexpr,
    K: tl.constexpr,
    checkpoint_group_size: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)

    comp_dtype = XQ_batch_ptr.type.element_ty
    accum_dtype = W1_checkpoints_ptr.type.element_ty

    K_F_F_stride = K * F * F
    K_F_stride = K * F
    CS_stride = CS

    F_F_offset = batch * NH * F_F_stride + head * F_F_stride + tl.arange(0, F)[:, None] * F + tl.arange(0, F)[None, :]
    F_offset = batch * NH * F_stride + head * F_stride + tl.arange(0, F)[None, :]
    norm_offset = head * F_stride + tl.arange(0, F)
    norm_store_offset = batch * NH * F_stride + head * F_stride + tl.arange(0, F)[None, :]

    ln_weight = tl.load(ttt_norm_weight_ptr + norm_offset)[None, :]
    ln_bias = tl.load(ttt_norm_bias_ptr + norm_offset)[None, :]

    # Load upstream gradients
    grad_L_W1_last = tl.load(grad_L_W1_last_ptr + F_F_offset)
    grad_L_b1_last = tl.load(grad_L_b1_last_ptr + F_offset)

    # Allocate stack for accumulated output gradients
    grad_L_ttt_norm_weight = tl.zeros((1, F), dtype=accum_dtype)
    grad_L_ttt_norm_bias = tl.zeros((1, F), dtype=accum_dtype)

    # Iterate over checkpoints in reverse
    for checkpoint_idx in range(K - 1, -1, -1):
        W1_checkpoint_offset = (
            batch * NH * K_F_F_stride
            + head * K_F_F_stride
            + checkpoint_idx * F_F_stride
            + tl.arange(0, F)[:, None] * F
            + tl.arange(0, F)[None, :]
        )
        b1_checkpoint_offset = (
            batch * NH * K_F_stride
            + head * K_F_stride
            + checkpoint_idx * F_stride
            + tl.arange(0, 1)[:, None] * F
            + tl.arange(0, F)[None, :]
        )

        W1_init = tl.load(W1_checkpoints_ptr + W1_checkpoint_offset)
        b1_init = tl.load(b1_checkpoints_ptr + b1_checkpoint_offset)

        # Forward over mini-batches in checkpoint group
        for mini_batch_idx_in_group in range(0, checkpoint_group_size):
            mini_batch_idx = checkpoint_idx * checkpoint_group_size + mini_batch_idx_in_group

            CS_F_offset = (
                batch * NH * NC * CS_F_stride
                + head * NC * CS_F_stride
                + mini_batch_idx * CS_F_stride
                + tl.arange(0, CS)[:, None] * F
                + tl.arange(0, F)[None, :]
            )
            CS_CS_offset = (
                batch * NH * NC * CS_CS_stride
                + head * NC * CS_CS_stride
                + mini_batch_idx * CS_CS_stride
                + tl.arange(0, CS)[:, None] * CS
                + tl.arange(0, CS)[None, :]
            )
            last_CS_offset = (
                batch * NH * NC * CS_CS_stride
                + head * NC * CS_CS_stride
                + mini_batch_idx * CS_CS_stride
                + (CS - 1) * CS
                + tl.arange(0, CS)[:, None]
            )

            XQ_mini_batch = tl.load(XQ_batch_ptr + CS_F_offset)
            XV_mini_batch = tl.load(XV_batch_ptr + CS_F_offset)
            XK_mini_batch = tl.load(XK_batch_ptr + CS_F_offset)
            eta_mini_batch = tl.load(eta_batch_ptr + CS_CS_offset)
            last_eta_mini_batch = tl.load(eta_batch_ptr + last_CS_offset)

            (
                XQW_mini_batch,
                W1_last,
                b1_last,
                x_hat_ln,
                std_ln,
                grad_l_wrt_Z1,
                Attn1,
                x_hat_fused,
                grad_x_hat_fused,
                grad_output_fused,
                std_fused,
            ) = ttt_linear_mini_batch_forward(
                W1_init,
                b1_init,
                ln_weight,
                ln_bias,
                XQ_mini_batch,
                XK_mini_batch,
                XV_mini_batch,
                eta_mini_batch,
                last_eta_mini_batch,
                CS,
                F,
                comp_dtype,
                accum_dtype,
            )

            G_CS_F_offset = (
                batch * NH * checkpoint_group_size * CS_F_stride
                + head * checkpoint_group_size * CS_F_stride
                + mini_batch_idx_in_group * CS_F_stride
                + tl.arange(0, CS)[:, None] * F
                + tl.arange(0, F)[None, :]
            )
            G_F_F_offset = (
                batch * NH * checkpoint_group_size * F_F_stride
                + head * checkpoint_group_size * F_F_stride
                + mini_batch_idx_in_group * F_F_stride
                + tl.arange(0, F)[:, None] * F
                + tl.arange(0, F)[None, :]
            )
            G_CS_CS_offset = (
                batch * NH * checkpoint_group_size * CS_CS_stride
                + head * checkpoint_group_size * CS_CS_stride
                + mini_batch_idx_in_group * CS_CS_stride
                + tl.arange(0, CS)[:, None] * CS
                + tl.arange(0, CS)[None, :]
            )
            G_F_offset = (
                batch * NH * checkpoint_group_size * F_stride
                + head * checkpoint_group_size * F_stride
                + mini_batch_idx_in_group * F_stride
                + tl.arange(0, 1)[:, None] * F
                + tl.arange(0, F)[None, :]
            )
            G_CS_offset = (
                batch * NH * checkpoint_group_size * CS_stride
                + head * checkpoint_group_size * CS_stride
                + mini_batch_idx_in_group * CS_stride
                + tl.arange(0, CS)[:, None]
                + tl.arange(0, 1)[None, :]
            )

            # Store intermediate values
            tl.store(XQW_mini_batch_group_ptr + G_CS_F_offset, XQW_mini_batch)
            tl.store(W1_init_group_ptr + G_F_F_offset, W1_init)
            tl.store(b1_init_group_ptr + G_F_offset, b1_init)
            tl.store(x_hat_ln_group_ptr + G_CS_F_offset, x_hat_ln)
            tl.store(std_ln_group_ptr + G_CS_offset, std_ln)
            tl.store(grad_l_wrt_Z1_group_ptr + G_CS_F_offset, grad_l_wrt_Z1)
            tl.store(Attn1_group_ptr + G_CS_CS_offset, Attn1)
            tl.store(x_hat_fused_group_ptr + G_CS_F_offset, x_hat_fused)
            tl.store(grad_x_hat_fused_group_ptr + G_CS_F_offset, grad_x_hat_fused)
            tl.store(grad_output_fused_group_ptr + G_CS_F_offset, grad_output_fused)
            tl.store(std_fused_group_ptr + G_CS_offset, std_fused)

            W1_init = W1_last
            b1_init = b1_last

        # Backward over mini-batches in checkpoint group in reverse
        for mini_batch_idx_in_group in range(checkpoint_group_size - 1, -1, -1):
            mini_batch_idx = checkpoint_idx * checkpoint_group_size + mini_batch_idx_in_group

            CS_F_offset = (
                batch * NH * NC * CS_F_stride
                + head * NC * CS_F_stride
                + mini_batch_idx * CS_F_stride
                + tl.arange(0, CS)[:, None] * F
                + tl.arange(0, F)[None, :]
            )
            CS_CS_offset = (
                batch * NH * NC * CS_CS_stride
                + head * NC * CS_CS_stride
                + mini_batch_idx * CS_CS_stride
                + tl.arange(0, CS)[:, None] * CS
                + tl.arange(0, CS)[None, :]
            )
            last_CS_offset = (
                batch * NH * NC * CS_CS_stride
                + head * NC * CS_CS_stride
                + mini_batch_idx * CS_CS_stride
                + (CS - 1) * CS
                + tl.arange(0, CS)[:, None]
            )

            G_CS_F_offset = (
                batch * NH * checkpoint_group_size * CS_F_stride
                + head * checkpoint_group_size * CS_F_stride
                + mini_batch_idx_in_group * CS_F_stride
                + tl.arange(0, CS)[:, None] * F
                + tl.arange(0, F)[None, :]
            )
            G_F_F_offset = (
                batch * NH * checkpoint_group_size * F_F_stride
                + head * checkpoint_group_size * F_F_stride
                + mini_batch_idx_in_group * F_F_stride
                + tl.arange(0, F)[:, None] * F
                + tl.arange(0, F)[None, :]
            )
            G_CS_CS_offset = (
                batch * NH * checkpoint_group_size * CS_CS_stride
                + head * checkpoint_group_size * CS_CS_stride
                + mini_batch_idx_in_group * CS_CS_stride
                + tl.arange(0, CS)[:, None] * CS
                + tl.arange(0, CS)[None, :]
            )
            G_F_offset = (
                batch * NH * checkpoint_group_size * F_stride
                + head * checkpoint_group_size * F_stride
                + mini_batch_idx_in_group * F_stride
                + tl.arange(0, 1)[:, None] * F
                + tl.arange(0, F)[None, :]
            )
            G_CS_offset = (
                batch * NH * checkpoint_group_size * CS_stride
                + head * checkpoint_group_size * CS_stride
                + mini_batch_idx_in_group * CS_stride
                + tl.arange(0, CS)[:, None]
                + tl.arange(0, 1)[None, :]
            )

            grad_L_XQW_mini_batch = tl.load(grad_L_XQW_mini_batch_ptr + CS_F_offset)

            XQ_mini_batch = tl.load(XQ_batch_ptr + CS_F_offset)
            XV_mini_batch = tl.load(XV_batch_ptr + CS_F_offset)
            XK_mini_batch = tl.load(XK_batch_ptr + CS_F_offset)
            eta_mini_batch = tl.load(eta_batch_ptr + CS_CS_offset)
            last_eta_mini_batch = tl.load(eta_batch_ptr + last_CS_offset)

            XQW_mini_batch = tl.load(XQW_mini_batch_group_ptr + G_CS_F_offset)
            W1_init = tl.load(W1_init_group_ptr + G_F_F_offset)
            b1_init = tl.load(b1_init_group_ptr + G_F_offset)
            x_hat_ln = tl.load(x_hat_ln_group_ptr + G_CS_F_offset)
            std_ln = tl.load(std_ln_group_ptr + G_CS_offset)
            grad_l_wrt_Z1 = tl.load(grad_l_wrt_Z1_group_ptr + G_CS_F_offset)
            Attn1 = tl.load(Attn1_group_ptr + G_CS_CS_offset)
            x_hat_fused = tl.load(x_hat_fused_group_ptr + G_CS_F_offset)
            grad_x_hat_fused = tl.load(grad_x_hat_fused_group_ptr + G_CS_F_offset)
            grad_output_fused = tl.load(grad_output_fused_group_ptr + G_CS_F_offset)
            std_fused = tl.load(std_fused_group_ptr + G_CS_offset)

            (
                grad_L_ttt_norm_weight_mini_batch,
                grad_L_ttt_norm_bias_mini_batch,
                grad_L_W1_init,
                grad_L_b1_init,
                grad_L_XQ_mini_batch,
                grad_L_XV_mini_batch,
                grad_L_XK_mini_batch,
                grad_L_eta_mini_batch,
            ) = ttt_linear_mini_batch_backward(
                # MatMul
                XQ_mini_batch,
                XK_mini_batch,
                W1_init,
                b1_init,
                # LnFusedL2BWD
                ln_weight,
                ln_bias,
                std_fused,
                x_hat_fused,
                grad_output_fused,
                grad_x_hat_fused,
                grad_l_wrt_Z1,
                # Dual Form
                Attn1,
                eta_mini_batch,
                last_eta_mini_batch,
                # LN
                std_ln,
                x_hat_ln,
                # Upstream gradients
                grad_L_W1_last,
                grad_L_b1_last,
                grad_L_XQW_mini_batch,
                # Strides
                CS_F_stride,
                F_F_stride,
                CS_CS_stride,
                F_stride,
                # Constant expressions
                NH,
                CS,
                F,
                comp_dtype,
                accum_dtype,
            )

            # Store mini-batch output gradients
            tl.store(grad_L_XQ_ptr + CS_F_offset, grad_L_XQ_mini_batch)
            tl.store(grad_L_XV_ptr + CS_F_offset, grad_L_XV_mini_batch)
            tl.store(grad_L_XK_ptr + CS_F_offset, grad_L_XK_mini_batch)
            tl.store(grad_L_eta_ptr + CS_CS_offset, grad_L_eta_mini_batch)

            # Accumulate / update output gradients
            grad_L_W1_last = grad_L_W1_init
            grad_L_b1_last = grad_L_b1_init
            grad_L_ttt_norm_weight += grad_L_ttt_norm_weight_mini_batch
            grad_L_ttt_norm_bias += grad_L_ttt_norm_bias_mini_batch

    tl.store(grad_L_ttt_norm_weight_ptr + norm_store_offset, grad_L_ttt_norm_weight)
    tl.store(grad_L_ttt_norm_bias_ptr + norm_store_offset, grad_L_ttt_norm_bias)
    tl.store(grad_L_W1_init_ptr + F_F_offset, grad_L_W1_last)
    tl.store(grad_L_b1_init_ptr + F_offset, grad_L_b1_last)
