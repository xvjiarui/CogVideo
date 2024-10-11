import torch
import triton
import triton.language as tl


@triton.jit
def gelu_triton(x):
    # Constants used in the GELU approximation
    sqrt_2_over_pi = tl.constexpr(0.79788456)
    coeff = sqrt_2_over_pi * (x + 0.044715 * x * x * x)

    # Compute GELU: 0.5 * x * (1 + tanh(coeff))
    tanh_approximation = (2 / (1 + tl.exp(-2 * coeff))) - 1

    gelu_out = 0.5 * x * (1.0 + tanh_approximation)

    return gelu_out


@triton.jit
def gelu_bwd_triton(x: tl.float32) -> tl.float32:
    # Constants used in the GELU derivative approximation
    sqrt_2_over_pi = tl.constexpr(0.79788456)
    coeff = sqrt_2_over_pi * x * (1 + 0.044715 * x * x)

    # Compute tanh component
    tanh_out = (2 / (1 + tl.exp(-2 * coeff))) - 1

    # Compute derivative: 0.5 * (1 + tanh_out) + 0.5 * x * (1 - tanh_out^2) * (sqrt_2_over_pi + 0.1070322243 * x * x)
    derivative = 0.5 * (1.0 + tanh_out) + 0.5 * x * (1.0 - tanh_out * tanh_out) * (
        sqrt_2_over_pi + 0.1070322243 * x * x
    )

    return derivative


@triton.jit
def ttt_mlp_stage_1(
    # Scan inputs
    ttt_norm_weight_ptr,
    ttt_norm_bias_ptr,
    W1_init_ptr,
    b1_init_ptr,
    W2_init_ptr,
    b2_init_ptr,
    XV_batch_ptr,
    XK_batch_ptr,
    # Ouputs
    grad_l_wrt_Z1_ptr,
    grad_l_wrt_Z2_ptr,
    X2_ptr,
    # Strides
    CS_F_stride: tl.constexpr,
    F_F4_stride: tl.constexpr,
    F_stride: tl.constexpr,
    F4_stride: tl.constexpr,
    # Constant expressions
    NH: tl.constexpr,
    NC: tl.constexpr,
    CS: tl.constexpr,
    F: tl.constexpr,
    i,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)

    comp_type = XV_batch_ptr.type.element_ty

    CS_F_offset = (
        batch * NH * NC * CS_F_stride
        + head * NC * CS_F_stride
        + i * CS_F_stride
        + tl.arange(0, CS)[:, None] * F
        + tl.arange(0, F)[None, :]
    )
    CS_F4_offset = (
        batch * NH * NC * CS_F_stride * 4
        + head * NC * CS_F_stride * 4
        + i * CS_F_stride
        + tl.arange(0, CS)[:, None] * F * 4
        + tl.arange(0, F * 4)[None, :]
    )
    CS_F_intermediate_offset = (
        batch * NH * CS_F_stride + head * CS_F_stride + tl.arange(0, CS)[:, None] * F + tl.arange(0, F)[None, :]
    )
    CS_F4_intermediate_offset = (
        batch * NH * CS_F_stride * 4
        + head * CS_F_stride * 4
        + tl.arange(0, CS)[:, None] * F * 4
        + tl.arange(0, F * 4)[None, :]
    )
    F_F4_offset = (
        batch * NH * F_F4_stride
        + head * F_F4_stride
        + tl.arange(0, F)[:, None] * F * 4
        + tl.arange(0, tl.constexpr(F * 4))[None, :]
    )
    F4_F_offset = (
        batch * NH * F_F4_stride
        + head * F_F4_stride
        + tl.arange(0, tl.constexpr(F * 4))[:, None] * F
        + tl.arange(0, F)[None, :]
    )
    F_offset = batch * NH * F_stride + head * F_stride + tl.arange(0, F)[None, :]
    F4_offset = batch * NH * F4_stride + head * F4_stride + tl.arange(0, tl.constexpr(F * 4))[None, :]
    norm_offset = head * F_stride + tl.arange(0, F)

    # Stage 1: MatMul
    XK_mini_batch = tl.load(XK_batch_ptr + CS_F_offset)
    W1_init = tl.load(W1_init_ptr + F_F4_offset)
    b1_init = tl.load(b1_init_ptr + F4_offset)
    Z1 = (tl.dot(XK_mini_batch, W1_init.to(comp_type), allow_tf32=True) + b1_init).to(comp_type)

    X2 = gelu_triton(Z1).to(comp_type)
    W2_init = tl.load(W2_init_ptr + F4_F_offset)
    b2_init = tl.load(b2_init_ptr + F_offset)
    Z2 = (tl.dot(X2, W2_init.to(comp_type), allow_tf32=True) + b2_init).to(comp_type)
    tl.store(X2_ptr + CS_F4_intermediate_offset, X2)

    XV_mini_batch = tl.load(XV_batch_ptr + CS_F_offset)
    reconstruction_target = XV_mini_batch - XK_mini_batch

    # Stage 2: LnFusedL2BWD
    mu_fused = (tl.sum(Z2, axis=1) / F)[:, None].to(comp_type)
    var_fused = (tl.sum((Z2 - mu_fused) * (Z2 - mu_fused), axis=1) / F)[:, None].to(comp_type)

    std_fused = tl.sqrt(var_fused + 1e-6).to(comp_type)
    x_hat_fused = ((Z2 - mu_fused) / std_fused).to(comp_type)

    ln_weight = tl.load(ttt_norm_weight_ptr + norm_offset)[None, :]
    ln_bias = tl.load(ttt_norm_bias_ptr + norm_offset)[None, :]
    y = ln_weight * x_hat_fused + ln_bias
    grad_output_fused = y - reconstruction_target
    grad_x_hat_fused = grad_output_fused * ln_weight

    grad_l_wrt_Z2 = (
        (1.0 / F)
        * (
            F * grad_x_hat_fused
            - tl.sum(grad_x_hat_fused, axis=1)[:, None]
            - x_hat_fused * tl.sum(grad_x_hat_fused * x_hat_fused, axis=1)[:, None]
        )
        / std_fused
    ).to(comp_type)
    tl.store(grad_l_wrt_Z2_ptr + CS_F_intermediate_offset, grad_l_wrt_Z2)

    grad_l_wrt_Z2_W2 = tl.dot(grad_l_wrt_Z2, tl.trans(W2_init).to(comp_type), allow_tf32=True).to(comp_type)
    grad_l_wrt_Z1 = grad_l_wrt_Z2_W2 * gelu_bwd_triton(Z1).to(comp_type)
    tl.store(grad_l_wrt_Z1_ptr + CS_F4_intermediate_offset, grad_l_wrt_Z1)


@triton.jit
def ttt_mlp_stage_2(
    # Scan inputs
    W1_init_ptr,
    b1_init_ptr,
    XQ_batch_ptr,
    XK_batch_ptr,
    eta_batch_ptr,
    # Intermediates
    grad_l_wrt_Z1_ptr,  # Output X2_bar here
    # Strides
    CS_F_stride: tl.constexpr,
    F_F4_stride: tl.constexpr,
    CS_CS_stride: tl.constexpr,
    F_stride: tl.constexpr,
    F4_stride: tl.constexpr,
    # Constant expressions
    NH: tl.constexpr,
    NC: tl.constexpr,
    CS: tl.constexpr,
    F: tl.constexpr,
    i,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)

    comp_type = XK_batch_ptr.type.element_ty

    CS_F_offset = (
        batch * NH * NC * CS_F_stride
        + head * NC * CS_F_stride
        + i * CS_F_stride
        + tl.arange(0, CS)[:, None] * F
        + tl.arange(0, F)[None, :]
    )
    CS_F4_intermediate_offset = (
        batch * NH * CS_F_stride * 4
        + head * CS_F_stride * 4
        + tl.arange(0, CS)[:, None] * F * 4
        + tl.arange(0, F * 4)[None, :]
    )
    CS_CS_offset = (
        batch * NH * NC * CS_CS_stride
        + head * NC * CS_CS_stride
        + i * CS_CS_stride
        + tl.arange(0, CS)[:, None] * CS
        + tl.arange(0, CS)[None, :]
    )
    last_CS_offset = (
        batch * NH * NC * CS_CS_stride
        + head * NC * CS_CS_stride
        + i * CS_CS_stride
        + (CS - 1) * CS
        + tl.arange(0, CS)[:, None]
    )
    F_F4_offset = (
        batch * NH * F_F4_stride
        + head * F_F4_stride
        + tl.arange(0, F)[:, None] * F * 4
        + tl.arange(0, tl.constexpr(F * 4))[None, :]
    )
    F4_offset = batch * NH * F4_stride + head * F4_stride + tl.arange(0, tl.constexpr(F * 4))[None, :]

    # Stage 3: Dual Form
    mask = tl.arange(0, CS)[:, None] >= tl.arange(0, CS)[None, :]
    last_eta_mini_batch = tl.load(eta_batch_ptr + last_CS_offset)

    W1_init = tl.load(W1_init_ptr + F_F4_offset)
    b1_init = tl.load(b1_init_ptr + F4_offset)
    grad_l_wrt_Z1 = tl.load(grad_l_wrt_Z1_ptr + CS_F4_intermediate_offset)
    XK_mini_batch = tl.load(XK_batch_ptr + CS_F_offset)

    W1_last = W1_init - tl.dot(tl.trans(last_eta_mini_batch * XK_mini_batch), grad_l_wrt_Z1, allow_tf32=True)
    b1_last = b1_init - tl.sum(last_eta_mini_batch * grad_l_wrt_Z1, axis=0)[None, :]
    tl.store(W1_init_ptr + F_F4_offset, W1_last)
    tl.store(b1_init_ptr + F4_offset, b1_last)

    eta_mini_batch = tl.load(eta_batch_ptr + CS_CS_offset)
    XQ_mini_batch = tl.load(XQ_batch_ptr + CS_F_offset)

    Attn1 = tl.where(mask, tl.dot(XQ_mini_batch, tl.trans(XK_mini_batch), allow_tf32=True), 0).to(comp_type)
    b1_bar = (b1_init - tl.dot(tl.where(mask, eta_mini_batch, 0).to(comp_type), grad_l_wrt_Z1, allow_tf32=True)).to(
        comp_type
    )
    Z1_bar = (
        tl.dot(XQ_mini_batch, W1_init.to(comp_type), allow_tf32=True)
        - tl.dot(eta_mini_batch * Attn1, grad_l_wrt_Z1, allow_tf32=True)
        + b1_bar
    ).to(comp_type)
    X2_bar = gelu_triton(Z1_bar).to(comp_type)
    tl.store(grad_l_wrt_Z1_ptr + CS_F4_intermediate_offset, X2_bar)


@triton.jit
def ttt_mlp_stage_3(
    # Scan inputs
    ttt_norm_weight_ptr,
    ttt_norm_bias_ptr,
    W2_init_ptr,
    b2_init_ptr,
    eta_batch_ptr,
    # Intermediates
    grad_l_wrt_Z2_ptr,
    X2_ptr,
    X2_bar_ptr,
    # Ouputs
    Z2_bar_ln_ptr,
    # Strides
    CS_F_stride: tl.constexpr,
    F_F4_stride: tl.constexpr,
    CS_CS_stride: tl.constexpr,
    F_stride: tl.constexpr,
    F4_stride: tl.constexpr,
    # Constant expressions
    NH: tl.constexpr,
    NC: tl.constexpr,
    CS: tl.constexpr,
    F: tl.constexpr,
    i,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)

    comp_type = eta_batch_ptr.type.element_ty

    CS_F_offset = (
        batch * NH * NC * CS_F_stride
        + head * NC * CS_F_stride
        + i * CS_F_stride
        + tl.arange(0, CS)[:, None] * F
        + tl.arange(0, F)[None, :]
    )
    CS_F_intermediate_offset = (
        batch * NH * CS_F_stride + head * CS_F_stride + tl.arange(0, CS)[:, None] * F + tl.arange(0, F)[None, :]
    )
    CS_F4_intermediate_offset = (
        batch * NH * CS_F_stride * 4
        + head * CS_F_stride * 4
        + tl.arange(0, CS)[:, None] * F * 4
        + tl.arange(0, F * 4)[None, :]
    )
    CS_CS_offset = (
        batch * NH * NC * CS_CS_stride
        + head * NC * CS_CS_stride
        + i * CS_CS_stride
        + tl.arange(0, CS)[:, None] * CS
        + tl.arange(0, CS)[None, :]
    )
    last_CS_offset = (
        batch * NH * NC * CS_CS_stride
        + head * NC * CS_CS_stride
        + i * CS_CS_stride
        + (CS - 1) * CS
        + tl.arange(0, CS)[:, None]
    )
    F4_F_offset = (
        batch * NH * F_F4_stride
        + head * F_F4_stride
        + tl.arange(0, tl.constexpr(F * 4))[:, None] * F
        + tl.arange(0, F)[None, :]
    )
    F_offset = batch * NH * F_stride + head * F_stride + tl.arange(0, F)[None, :]
    norm_offset = head * F_stride + tl.arange(0, F)

    mask = tl.arange(0, CS)[:, None] >= tl.arange(0, CS)[None, :]
    last_eta_mini_batch = tl.load(eta_batch_ptr + last_CS_offset)

    W2_init = tl.load(W2_init_ptr + F4_F_offset)
    b2_init = tl.load(b2_init_ptr + F_offset)

    grad_l_wrt_Z2 = tl.load(grad_l_wrt_Z2_ptr + CS_F_intermediate_offset)
    X2 = tl.load(X2_ptr + CS_F4_intermediate_offset)

    W2_last = W2_init - tl.dot(tl.trans(last_eta_mini_batch * X2), grad_l_wrt_Z2, allow_tf32=True)
    b2_last = b2_init - tl.sum(last_eta_mini_batch * grad_l_wrt_Z2, axis=0)[None, :]
    tl.store(W2_init_ptr + F4_F_offset, W2_last)
    tl.store(b2_init_ptr + F_offset, b2_last)

    X2_bar = tl.load(X2_bar_ptr + CS_F4_intermediate_offset)
    eta_mini_batch = tl.load(eta_batch_ptr + CS_CS_offset)

    Attn2 = tl.where(mask, tl.dot(X2_bar, tl.trans(X2), allow_tf32=True), 0).to(comp_type)
    b2_bar = (b2_init - tl.dot(tl.where(mask, eta_mini_batch, 0).to(comp_type), grad_l_wrt_Z2, allow_tf32=True)).to(
        comp_type
    )
    Z2_bar = (
        tl.dot(X2_bar, W2_init.to(comp_type), allow_tf32=True)
        - tl.dot(eta_mini_batch * Attn2, grad_l_wrt_Z2, allow_tf32=True)
        + b2_bar
    ).to(comp_type)

    # Stage 4: LN
    mu_ln = (tl.sum(Z2_bar, axis=1)[:, None] / F).to(comp_type)
    var_ln = (tl.sum((Z2_bar - mu_ln) * (Z2_bar - mu_ln), axis=1)[:, None] / F).to(comp_type)
    std_ln = tl.sqrt(var_ln + 1e-6).to(comp_type)
    x_hat_ln = ((Z2_bar - mu_ln) / std_ln).to(comp_type)

    ln_weight = tl.load(ttt_norm_weight_ptr + norm_offset)[None, :]
    ln_bias = tl.load(ttt_norm_bias_ptr + norm_offset)[None, :]

    Z2_bar_ln = ln_weight * x_hat_ln + ln_bias
    tl.store(Z2_bar_ln_ptr + CS_F_offset, Z2_bar_ln)
