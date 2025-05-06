# src/distributed_xfold/xsmm_kernels/prototypes/register_fake_kernels.py
import torch

try:
    import tpp_pytorch_extension._C
except ImportError as e:
    print(f"Warning: Could not import tpp_pytorch_extension._C. FakeTensor kernels will not be registered: {e}")

from . import TRI_BLOCKSIZE, QKV_BLOCKSIZE

# 库名::算子名 格式
@torch.library.register_fake("_alpha_attention::grid_self_attn_forward")
def _fake_grid_self_attn_forward(
    q_data, bias, nonbatched_bias,
    query_w, key_w, value_w, gating_w, output_w,
    key_dim, value_dim
):
    B_t = q_data.shape[0]
    Sp_t = q_data.shape[1] # Original sequence length
    HS_t = q_data.shape[2] # Channel dimension

    S_t = Sp_t
    if Sp_t % QKV_BLOCKSIZE != 0:
        S_t = (Sp_t // QKV_BLOCKSIZE + 1) * QKV_BLOCKSIZE

    padded_output_shape = (B_t, S_t, HS_t)
    output_dtype = q_data.dtype
    output_device = q_data.device

    padded_output = torch.empty(padded_output_shape, dtype=output_dtype, device=output_device)
    return padded_output

@torch.library.register_fake("_alpha_attention::transition_forward")
def _fake_grid_transition_forward(
    act, transition1, transition2, 
    layernorm_weight, layernorm_bias
):
    return torch.empty(act.shape, dtype=act.dtype, device=act.device)

@torch.library.register_fake("_alpha_attention::batch_diffusion_self_attention_forward")
def _fake_batch_diffusion_self_attention_forward(
    q_data, bias, nonbatched_bias,
    query_w, query_b, key_w, value_w, gating_w,
    key_dim, value_dim
):
    B_t = q_data.shape[0]
    Sp_t = q_data.shape[1]

    N_t = query_w.shape[-2]
    H_t = query_w.shape[-1]

    S_t = Sp_t
    if Sp_t % QKV_BLOCKSIZE != 0:
        S_t = (Sp_t // QKV_BLOCKSIZE + 1) * QKV_BLOCKSIZE

    padded_output_shape = (B_t, S_t, N_t, H_t)
    output_dtype = q_data.dtype
    output_device = q_data.device

    padded_output = torch.empty(padded_output_shape, dtype=output_dtype, device=output_device)
    return padded_output

# @torch.library.register_fake("_alpha_attention::diffusion_self_attention_forward")
# def _fake_diffusion_self_attention_forward(
#     q_data, bias, nonbatched_bias,
#     query_w, query_b, key_w, value_w, gating_w,
#     key_dim, value_dim
# ):
#     Sp_t = q_data.shape[0]
#     N_t = query_w.shape[-2]
#     H_t = query_w.shape[-1]

#     output_shape = (Sp_t, N_t, H_t)
#     output_dtype = q_data.dtype
#     output_device = q_data.device

#     return torch.empty(output_shape, dtype=output_dtype, device=output_device)

@torch.library.register_fake("_alpha_attention::batch_diffusion_cross_attention_forward")
def _fake_batch_diffusion_cross_attention_forward(
    q_data, m_data, batched_bias,
    query_w, query_b, key_w, value_w, gating_w,
    key_dim, value_dim
):
    B_t = q_data.shape[0]
    Sp_t = q_data.shape[1]

    N_t = query_w.shape[-2]
    H_t = query_w.shape[-1]

    S_t = Sp_t
    if Sp_t % QKV_BLOCKSIZE != 0:
        S_t = (Sp_t // QKV_BLOCKSIZE + 1) * QKV_BLOCKSIZE

    padded_output_shape = (B_t, S_t, N_t, H_t)
    output_dtype = q_data.dtype
    output_device = q_data.device

    padded_output = torch.empty(padded_output_shape, dtype=output_dtype, device=output_device)
    return padded_output


@torch.library.register_fake("_alpha_attention::traingle_multiplication_forward")
def _fake_traingle_multiplication_forward(
    act, mask, equation_flag,
    left_norm_input_weight, left_norm_input_bias, projection_weight,
    gate_weight, center_norm_weight, center_norm_bias,
    output_projection_weight, gating_linear_weight
):
    Bp_t = act.shape[0]
    Sp_t = act.shape[1] # Original sequence length
    act_dim = act.shape[2] # Channel dimension

    S_t = Sp_t
    B_t = Bp_t
    if Bp_t % TRI_BLOCKSIZE != 0:
        B_t = (Bp_t // TRI_BLOCKSIZE + 1) * TRI_BLOCKSIZE
    if Sp_t % TRI_BLOCKSIZE != 0:
        S_t = (Sp_t // TRI_BLOCKSIZE + 1) * TRI_BLOCKSIZE

    padded_output_shape = (B_t, S_t, act_dim)
    output_dtype = act.dtype
    output_device = act.device

    padded_output = torch.empty(padded_output_shape, dtype=output_dtype, device=output_device)
    return padded_output