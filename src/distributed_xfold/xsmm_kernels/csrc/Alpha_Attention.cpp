/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Narendra Chaudhary (Intel Corp.)
 ******************************************************************************/

#include <omp.h>
#include <stdio.h>
#include <torch/extension.h>
#include <cmath>
#include <iostream>
#include <tuple>

#include <ATen/record_function.h>
#include "ext_tpp.h"
#include "init.h"
#include "timing.h"
#include "xsmm_functors.h"

using namespace tpp;
#include "tensor_helper.h"

#define TRI_BLOCKSIZE 32

#define QKV_BLOCKSIZE 64
#define A_BLOCKSIZE 64
#define Ak_BLOCKSIZE 512
#define C_BLOCKSIZE 64
#define T_BLOCKSIZE 128

REGISTER_SCOPE(alpha_q_gemm, "alpha_q_gemm");
REGISTER_SCOPE(alpha_k_gemm, "alpha_k_gemm");
REGISTER_SCOPE(alpha_v_gemm, "alpha_v_gemm");

REGISTER_SCOPE(alpha_ac_gemm, "alpha_ac_gemm");
REGISTER_SCOPE(alpha_o_gemm, "alpha_o_gemm");
REGISTER_SCOPE(alpha_t_gemm, "alpha_t_gemm");

REGISTER_SCOPE(proj_gemm, "proj_gemm");
REGISTER_SCOPE(out_gemm, "out_gemm");
REGISTER_SCOPE(gate_gemm, "gate_gemm");
REGISTER_SCOPE(eq_bmm, "eq_gemm");
REGISTER_SCOPE(layer_norm_input, "layer_norm_input");
REGISTER_SCOPE(left_norm_input, "left_norm_input");

at::Tensor grid_self_attention_fwd(
    at::Tensor& q_data,
    // at::Tensor& m_data,
    at::Tensor& bias,
    at::Tensor& nonbatched_bias,
    at::Tensor& query_w,
    at::Tensor& key_w,
    at::Tensor& value_w,
    at::Tensor& gating_w,
    at::Tensor& output_w,
    int64_t key_dim,
    int64_t value_dim) {
  GlobalPass _gp(FWD);
if (q_data.dtype() == at::kFloat) {
    typedef float T;
#include "grid_self_attention.h"
  } else {
    typedef bfloat16 T;
#include "grid_self_attention_bf16.h"
  }
}

at::Tensor batch_diffusion_self_attention_fwd(
    at::Tensor& q_data,
    // at::Tensor& bias,
    at::Tensor& bias,
    at::Tensor& nonbatched_bias,
    at::Tensor& query_w,
    at::Tensor& query_b,
    at::Tensor& key_w,
    at::Tensor& value_w,
    at::Tensor& gating_w,
    int64_t key_dim,
    int64_t value_dim) {
  GlobalPass _gp(FWD);
if (q_data.dtype() == at::kFloat) {
  typedef float T;
#include "batch_diffusion_self_attention.h"
  } else {
    typedef bfloat16 T;
#include "batch_diffusion_self_attention_bf16.h"
  }
}

// at::Tensor diffusion_self_attention_fwd(
//     at::Tensor& q_data,
//     // at::Tensor& bias,
//     at::Tensor& bias,
//     at::Tensor& nonbatched_bias,
//     at::Tensor& query_w,
//     at::Tensor& query_b,
//     at::Tensor& key_w,
//     at::Tensor& value_w,
//     at::Tensor& gating_w,
//     int64_t key_dim,
//     int64_t value_dim) {
//   GlobalPass _gp(FWD);
// if (q_data.dtype() == at::kFloat) {
//   typedef float T;
// #include "diffusion_self_attention.h"
//   } else {
//     typedef bfloat16 T;
// #include "diffusion_self_attention_bf16.h"
//   }
// }

at::Tensor transition_fwd(
    at::Tensor& act,
    at::Tensor& transition1,
    at::Tensor& transition2,
    at::Tensor& layernorm_weight,
    at::Tensor& layernorm_bias) {
  GlobalPass _gp(FWD);
  typedef float T;
  #include "transition.h"
}

at::Tensor batch_diffusion_cross_attention_fwd(
    at::Tensor& q_data,
    at::Tensor& m_data,
    at::Tensor& batched_bias,
    at::Tensor& query_w,
    at::Tensor& query_b,
    at::Tensor& key_w,
    at::Tensor& value_w,
    at::Tensor& gating_w,
    int64_t key_dim,
    int64_t value_dim) {
  GlobalPass _gp(FWD);
if (q_data.dtype() == at::kFloat) {
    typedef float T;
#include "batch_diffusion_cross_attention.h"
  } else {
    typedef bfloat16 T;
#include "batch_diffusion_cross_attention_bf16.h"
  }
}

at::Tensor traingle_multiplication_forward(
    at::Tensor& act,
    at::Tensor& mask,
    int64_t equation_flag,
    at::Tensor& left_norm_input_weight,
    at::Tensor& left_norm_input_bias,
    at::Tensor& projection_weight,
    at::Tensor& gate_weight,
    at::Tensor& center_norm_weight,
    at::Tensor& center_norm_bias,
    at::Tensor& output_projection_weight,
    at::Tensor& gating_linear_weight
    ) {
  GlobalPass _gp(FWD);
if (act.dtype() == at::kFloat) {
    typedef float T;
#include "triangle_multiplication.h"
  } else {
    typedef bfloat16 T;
#include "triangle_multiplication_bf16.h"
  }
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
// REGISTER_SUBMODULE(_alpha_attention, m) {
//   m.def("grid_self_attn_forward", &grid_self_attention_fwd, "Grid Self attention forward");
//   m.def("batch_diffusion_self_attention_forward", &batch_diffusion_self_attention_fwd, "Diffusion Self Attention with batch dim");
//   m.def("batch_diffusion_cross_attention_forward", &batch_diffusion_cross_attention_fwd, "Diffusion Attention with batch dim");
//   m.def("traingle_multiplication_forward", &traingle_multiplication_forward, "Traingle Multiplication forward");
// }


// Define the operator schemas in the "_alpha_attention" library
TORCH_LIBRARY(_alpha_attention, m) {
  m.def("grid_self_attn_forward("
        "Tensor q_data, Tensor bias, Tensor nonbatched_bias, "
        "Tensor query_w, Tensor key_w, Tensor value_w, Tensor gating_w, Tensor output_w, "
        "int key_dim, int value_dim) -> Tensor");

  m.def("transition_forward("
        "Tensor act, Tensor transition1, Tensor transition2, "
        "Tensor layernorm_weight, Tensor layernorm_bias) -> Tensor"); // Note: Query_b added based on the C++ signature
  
  m.def("batch_diffusion_self_attention_forward("
        "Tensor q_data, Tensor bias, Tensor nonbatched_bias, "
        "Tensor query_w, Tensor query_b, Tensor key_w, Tensor value_w, Tensor gating_w, "
        "int key_dim, int value_dim) -> Tensor"); // Note: Query_b added based on the C++ signature

  m.def("diffusion_self_attention_forward("
        "Tensor q_data, Tensor bias, Tensor nonbatched_bias, "
        "Tensor query_w, Tensor query_b, Tensor key_w, Tensor value_w, Tensor gating_w, "
        "int key_dim, int value_dim) -> Tensor"); // Note: Query_b added based on the C++ signature

  m.def("batch_diffusion_cross_attention_forward("
        "Tensor q_data, Tensor m_data, Tensor batched_bias, "
        "Tensor query_w, Tensor query_b, Tensor key_w, Tensor value_w, Tensor gating_w, "
        "int key_dim, int value_dim) -> Tensor"); // Note: Query_b added based on the C++ signature

  m.def("traingle_multiplication_forward("
        "Tensor act, Tensor mask, int equation_flag, "
        "Tensor left_norm_input_weight, Tensor left_norm_input_bias, Tensor projection_weight, "
        "Tensor gate_weight, Tensor center_norm_weight, Tensor center_norm_bias, "
        "Tensor output_projection_weight, Tensor gating_linear_weight) -> Tensor");
}

// Provide CPU implementations for the operators in the "_alpha_attention" library
// The library name in TORCH_LIBRARY_IMPL must match the one in TORCH_LIBRARY
TORCH_LIBRARY_IMPL(_alpha_attention, CPU, m) {
  m.impl("grid_self_attn_forward", &grid_self_attention_fwd);
  m.impl("transition_forward", &transition_fwd);
  m.impl("batch_diffusion_self_attention_forward", &batch_diffusion_self_attention_fwd);
  // m.impl("diffusion_self_attention_forward", &diffusion_self_attention_fwd);
  m.impl("batch_diffusion_cross_attention_forward", &batch_diffusion_cross_attention_fwd);
  m.impl("traingle_multiplication_forward", &traingle_multiplication_forward);
}