/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Narendra Chaudhary (Intel Corp.)
 ******************************************************************************/

RECORD_FUNCTION(
    "Gating attention forward",
    std::vector<c10::IValue>({q_data})); // For recording time

int64_t B_t = q_data.size(0); /* Batch (512) */
int64_t Sp_t = q_data.size(1); /* Query (764) */
int64_t HS_t = q_data.size(2); /* Channels (256) */

int64_t N_t = query_w.size(1); /* number of heads (8) */
int64_t H_t = query_w.size(2); /* head size (32) */

auto flag = nonbatched_bias.size(0) > 0;

int64_t S_t = Sp_t;
if (Sp_t % QKV_BLOCKSIZE != 0) {
  S_t = (Sp_t / QKV_BLOCKSIZE + 1) * QKV_BLOCKSIZE; // 768

  auto q_data_pad = q_data.new_zeros({B_t, S_t - Sp_t, HS_t});
  auto bias_pad = bias.new_zeros({B_t, 1, 1, S_t - Sp_t});
  auto nonbatched_bias_pad1 =
      nonbatched_bias.new_zeros({N_t, Sp_t, S_t - Sp_t});
  auto nonbatched_bias_pad2 = nonbatched_bias.new_zeros({N_t, S_t - Sp_t, S_t});

  q_data = at::cat({q_data, q_data_pad}, 1);
  bias = at::cat({bias, bias_pad}, 3);
  if (flag) {
    nonbatched_bias = at::cat({nonbatched_bias, nonbatched_bias_pad1}, 2);
    nonbatched_bias = at::cat({nonbatched_bias, nonbatched_bias_pad2}, 1);
  }
}

bias = bias.contiguous();
nonbatched_bias = nonbatched_bias.contiguous();
auto sfmask = -30000 * q_data.new_ones(S_t - Sp_t);
auto sfmask_a = GetVLAPtr<T>(sfmask, {1L});

auto q_data_a = GetVLAPtr<T>(q_data, {S_t, HS_t});
auto bias_a = GetVLAPtr<T>(bias, {S_t});
auto nonbatched_bias_a = GetVLAPtr<T>(nonbatched_bias, {N_t, S_t, S_t});

auto query_w_a = GetVLAPtr<T>(query_w, {N_t, H_t});
auto query_b_a = GetVLAPtr<T>(query_b, {N_t * H_t});

auto key_w_a = GetVLAPtr<T>(key_w, {N_t, H_t});
auto value_w_a = GetVLAPtr<T>(value_w, {N_t, H_t});
auto gating_w_a = GetVLAPtr<T>(gating_w, {N_t, H_t});

auto q = q_data.new_empty({B_t, S_t, N_t, H_t}); /* [512, 764, 8, 32] */
auto q_a = GetVLAPtr<T>(q, {S_t, N_t, H_t});

auto k = q_data.new_empty({B_t, S_t* N_t* H_t}); /* [512, 764, 8, 32] */
auto k_a = GetVLAPtr<T>(k, {S_t * N_t * H_t});

auto v = q_data.new_empty({B_t, S_t, N_t, H_t}); /* [512, 764, 8, 32] */
auto v_a = GetVLAPtr<T>(v, {S_t, N_t, H_t});

auto weighted_avg =
    q_data.new_empty({B_t, S_t, N_t, H_t}); /* [512, 764, 8, 32] */
auto weighted_avg_a = GetVLAPtr<T>(weighted_avg, {S_t, N_t, H_t});

auto output = q_data.new_empty({B_t, S_t, N_t, H_t}); /* [512, 764, 256] */
auto output_a = GetVLAPtr<T>(output, {S_t, N_t, H_t});

int64_t lda = HS_t;
int64_t ldb = N_t * H_t;
int64_t ldc = N_t * H_t;

auto qkv_brgemm_tpp = SCOPEITGEMM(
    (BrgemmTPP<
        T,
        T>(QKV_BLOCKSIZE, N_t* H_t, HS_t, 0, 0, lda, ldb, ldc, 0.0, 0, 1)));
auto q_addbias_tpp = SCOPEIT(AddBiasTPP<T>(QKV_BLOCKSIZE, HS_t, HS_t), BIAS);

auto k_trans_tpp = SCOPEIT(
    XformExtTPP<T>(
        QKV_BLOCKSIZE,
        N_t* H_t,
        N_t* H_t,
        QKV_BLOCKSIZE,
        N_t* H_t,
        S_t,
        XformTPP::XFORM_XPOSE_TPP),
    XPOSE);

auto scale_tpp = SCOPEIT((ScaleTPP<T, T>(QKV_BLOCKSIZE * HS_t)), EW_SCL);
auto zero_tpp = SCOPEIT(SetZeroTPP<T>(QKV_BLOCKSIZE * HS_t), EW_ZERO);
float alpha = (1.0 / sqrt(key_dim));


// auto q = at::mul(at::einsum("bqa,ahc->bqhc", {q_data, query_w}),
// (1.0/sqrt(key_dim))) ;     /* [512, 764, 8, 32]  = [512, 764, 256] * [256, 8,
// 32] */

{
  RECORD_SCOPE(alpha_q_gemm, {q, q_data, query_w, query_b});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
    for (int64_t i = 0; i < B_t; i++) {
      for (int64_t j = 0; j < S_t; j += QKV_BLOCKSIZE) {
        T tmp[QKV_BLOCKSIZE * N_t * H_t];
        qkv_brgemm_tpp(&q_data_a[i][j][0], &query_w_a[0][0][0], &tmp[0], 1);
        q_addbias_tpp(&query_b_a[0][0], &tmp[0]);
        scale_tpp(&tmp[0], &q_a[i][j][0][0], alpha);
      }
    }
  }
}

// auto k = at::einsum("bka,ahc->bkhc", {q_data, key_w}); /* [512, 764, 8, 32]
// = [512, 764, 256] * [256, 8, 32] */

{
  RECORD_SCOPE(alpha_k_gemm, {k, q_data, key_w});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
    for (int64_t i = 0; i < B_t; i++) {
      for (int64_t j = 0; j < S_t; j += QKV_BLOCKSIZE) {
        T tmp[QKV_BLOCKSIZE * N_t * H_t];
        qkv_brgemm_tpp(&q_data_a[i][j][0], &key_w_a[0][0][0], &tmp[0], 1);
        k_trans_tpp(&tmp[0], &k_a[i][j]); // [ 0*H_t*S_t + 0*S_t + j]
      }
    }
  }
}

// auto v = at::einsum("bka,ahc->bkhc", {q_data, value_w}); /* [512, 764, 8, 32]
// = [512, 764, 256] * [256, 8, 32] */
{
  RECORD_SCOPE(alpha_v_gemm, {v, q_data, value_w});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
    for (int64_t i = 0; i < B_t; i++) {
      for (int64_t j = 0; j < S_t; j += QKV_BLOCKSIZE) {
        qkv_brgemm_tpp(
            &q_data_a[i][j][0], &value_w_a[0][0][0], &v_a[i][j][0][0], 1);
      }
    }
  }
}

lda = H_t;
ldb = A_BLOCKSIZE;
ldc = S_t;

// logits = at::add(at::einsum("bqhc,bkhc->bhqk", {q, k}), bias); /* [512, 8,
// 764, 764]  = [512, 764, 8, 32] * [512, 764, 8, 32] + [512, 1, 1, 764] */ if
// (nonbatched_bias.size(0) > 0)
//     logits = at::add(logits, at::unsqueeze(nonbatched_bias, 0)); /* [512, 8,
//     764, 764]  = [512, 8, 764, 764] + [1, 8, 764, 764] */
// weights = at::_softmax(logits, -1, false); /* [512, 8, 764, 764] = [512, 8,
// 764, 764] */ auto weighted_avg = at::einsum("bhqk,bkhc->bqhc", {weights,
// v}).contiguous();          /* [512, 764, 8, 32]  = [512, 8, 764, 764] * [512,
// 764, 8, 32] */

auto a_zero_tpp = SCOPEIT(SetZeroTPP<T>(A_BLOCKSIZE * H_t), EW_ZERO);
auto a_cpy_tpp = SCOPEIT(CpyTPP<T>(A_BLOCKSIZE, H_t, H_t, N_t* H_t), EW_COPY);

if (S_t < 2048) {
  auto a_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, T>(
      A_BLOCKSIZE, A_BLOCKSIZE, H_t, 0, 0, N_t * H_t, S_t, S_t, 0.0, 0, 1)));

  auto c_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, T>(
      A_BLOCKSIZE,
      H_t,
      A_BLOCKSIZE,
      A_BLOCKSIZE,
      A_BLOCKSIZE * N_t * H_t,
      S_t,
      N_t * H_t,
      H_t,
      0.0,
      0,
      1)));

  auto a_addbias_tpp = SCOPEIT(AddBiasTPP<T>(A_BLOCKSIZE, S_t, S_t), BIAS);
  auto a_add_nbbias_tpp =
      SCOPEIT((AddTPP<T, T>(A_BLOCKSIZE, S_t, S_t, S_t)), BIAS);

  auto a_add_sfmask_tpp =
      SCOPEIT(AddBiasTPP<T>(A_BLOCKSIZE, S_t - Sp_t, S_t), BIAS);
  auto a_softmax_tpp =
      SCOPEIT((VarSoftMaxFwdTPP<float, T>(A_BLOCKSIZE, S_t)), SOFTMAX);

  {
    RECORD_SCOPE(alpha_ac_gemm, {q, k, bias});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(3)
      for (int64_t i = 0; i < B_t; i++) {
        for (int64_t n = 0; n < N_t; n++) {
          for (int64_t j1 = 0; j1 < S_t; j1 += A_BLOCKSIZE) {
            T tmp_o[A_BLOCKSIZE * H_t];
            T tmp_logits[A_BLOCKSIZE][S_t];

            for (int64_t j2 = 0; j2 < S_t; j2 += A_BLOCKSIZE) {
              a_brgemm_tpp(
                  &q_a[i][j1][n][0],
                  &k_a[i][n * H_t * S_t + j2],
                  &tmp_logits[0][j2],
                  1);
            }

            a_addbias_tpp(&bias_a[i][0], &tmp_logits[0][0]);
            if (flag)
              a_add_nbbias_tpp(
                  &nonbatched_bias_a[0][n][j1][0],
                  &tmp_logits[0][0],
                  &tmp_logits[0][0]);

            if (S_t == Sp_t) {
              a_softmax_tpp(1, &tmp_logits[0][0], &tmp_logits[0][0]);
            } else {
              a_add_sfmask_tpp(&sfmask_a[0][0], &tmp_logits[0][Sp_t]);
              a_softmax_tpp(1, &tmp_logits[0][0], &tmp_logits[0][0]);
            }

            c_brgemm_tpp(
                &tmp_logits[0][0],
                &v_a[i][0][n][0],
                &tmp_o[0],
                S_t / A_BLOCKSIZE);
            a_cpy_tpp(&tmp_o[0], &weighted_avg_a[i][j1][n][0]);
          }
        }
      }
    }
  }
} else {
  auto a_cpy2_tpp = SCOPEIT(CpyTPP<T>(A_BLOCKSIZE, H_t), EW_COPY);

  auto a_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, T>(
      A_BLOCKSIZE,
      Ak_BLOCKSIZE,
      H_t,
      0,
      0,
      N_t * H_t,
      S_t,
      Ak_BLOCKSIZE,
      0.0,
      0,
      1)));

  auto c_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, T>(
      A_BLOCKSIZE,
      H_t,
      Ak_BLOCKSIZE,
      0,
      0,
      Ak_BLOCKSIZE,
      N_t * H_t,
      H_t,
      0.0,
      0,
      1)));

  auto a_addbias_online_tpp =
      SCOPEIT(AddBiasTPP<T>(A_BLOCKSIZE, Ak_BLOCKSIZE, Ak_BLOCKSIZE), BIAS);
  auto a_add_nbbias_online_tpp = SCOPEIT(
      (AddTPP<T, T>(
          A_BLOCKSIZE, Ak_BLOCKSIZE, S_t, Ak_BLOCKSIZE, Ak_BLOCKSIZE)),
      BIAS);

  auto a_softmax_online_tpp = SCOPEIT(
      (VarSoftMaxFwdTPP<float, T>(A_BLOCKSIZE, Ak_BLOCKSIZE, true)), SOFTMAX);
  auto a_softmax_fixup_online =
      SCOPEIT(SoftMaxFixUpTPP<T>(A_BLOCKSIZE, H_t, true), EW_RCP);
  auto a_softmax_scale_online =
      SCOPEIT(SoftMaxFlashScaleTPP<T>(A_BLOCKSIZE, H_t, true), EW_RCP);

  // if (S_t % Ak_BLOCKSIZE != 0){
  int64_t lastBlockSize = S_t - (S_t / Ak_BLOCKSIZE) * Ak_BLOCKSIZE;
  if (lastBlockSize == 0)
    lastBlockSize = Ak_BLOCKSIZE; // handling the zero case
  auto a_brgemm_edge_tpp = SCOPEITGEMM((BrgemmTPP<T, T>(
      A_BLOCKSIZE,
      lastBlockSize,
      H_t,
      0,
      0,
      N_t * H_t,
      S_t,
      lastBlockSize,
      0.0,
      0,
      1)));

  auto c_brgemm_edge_tpp = SCOPEITGEMM((BrgemmTPP<T, T>(
      A_BLOCKSIZE,
      H_t,
      lastBlockSize,
      0,
      0,
      lastBlockSize,
      N_t * H_t,
      H_t,
      0.0,
      0,
      1)));

  auto a_addbias_online_edge_tpp =
      SCOPEIT(AddBiasTPP<T>(A_BLOCKSIZE, lastBlockSize, lastBlockSize), BIAS);
  auto a_add_nbbias_online_edge_tpp = SCOPEIT(
      (AddTPP<T, T>(
          A_BLOCKSIZE, lastBlockSize, S_t, lastBlockSize, lastBlockSize)),
      BIAS);

  auto a_add_sfmask_online_tpp =
      SCOPEIT(AddBiasTPP<T>(A_BLOCKSIZE, S_t - Sp_t, lastBlockSize), BIAS);

  auto a_softmax_online_edge_tpp = SCOPEIT(
      (VarSoftMaxFwdTPP<float, T>(A_BLOCKSIZE, lastBlockSize, true)), SOFTMAX);
  // }

  {
    RECORD_SCOPE(alpha_ac_gemm, {q, k, bias});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(3)
      for (int64_t i = 0; i < B_t; i++) {
        for (int64_t n = 0; n < N_t; n++) {
          for (int64_t j1 = 0; j1 < S_t; j1 += A_BLOCKSIZE) {
            T tmp_o1[A_BLOCKSIZE * H_t];
            T tmp_o2[A_BLOCKSIZE * H_t];
            T tmp_S[A_BLOCKSIZE * Ak_BLOCKSIZE];
            float omax[A_BLOCKSIZE], osum[A_BLOCKSIZE], cmax[A_BLOCKSIZE],
                csum[A_BLOCKSIZE];

            for (int64_t j2 = 0; j2 < (S_t / Ak_BLOCKSIZE) * Ak_BLOCKSIZE;
                 j2 += Ak_BLOCKSIZE) {
              a_brgemm_tpp(
                  &q_a[i][j1][n][0], &k_a[i][n * H_t * S_t + j2], tmp_S, 1);

              a_addbias_online_tpp(&bias_a[i][j2], tmp_S);
              if (flag)
                a_add_nbbias_online_tpp(
                    &nonbatched_bias_a[0][n][j1][j2], tmp_S, tmp_S);

              if (j2 == 0) {
                a_softmax_online_tpp(1, tmp_S, tmp_S, omax, osum, nullptr);
              } else {
                a_softmax_online_tpp(1, tmp_S, tmp_S, cmax, csum, omax);
              }

              c_brgemm_tpp(tmp_S, &v_a[i][j2][n][0], tmp_o1,
                           1); // O = P*V
              if (j2 == 0) {
                a_cpy2_tpp(tmp_o1, tmp_o2);
              } else {
                a_softmax_fixup_online(tmp_o1, tmp_o2, cmax, csum, omax, osum);
              }
            }

            if (S_t % Ak_BLOCKSIZE != 0) {
              T* tmp_S_edge = new T[A_BLOCKSIZE * lastBlockSize];
              int64_t j2 = (S_t / Ak_BLOCKSIZE) * Ak_BLOCKSIZE;
              a_brgemm_edge_tpp(
                  &q_a[i][j1][n][0],
                  &k_a[i][n * H_t * S_t + j2],
                  tmp_S_edge,
                  1);

              a_addbias_online_edge_tpp(&bias_a[i][j2], tmp_S_edge);
              if (flag) {
                a_add_nbbias_online_edge_tpp(
                    &nonbatched_bias_a[0][n][j1][j2], tmp_S_edge, tmp_S_edge);
              }

              a_add_sfmask_online_tpp(&sfmask_a[0][0], &tmp_S_edge[Sp_t - j2]);
              a_softmax_online_edge_tpp(
                  1, tmp_S_edge, tmp_S_edge, cmax, csum, omax);

              c_brgemm_edge_tpp(tmp_S_edge, &v_a[i][j2][n][0], tmp_o1, 1);
              a_softmax_fixup_online(tmp_o1, tmp_o2, cmax, csum, omax, osum);
              delete[] tmp_S_edge;
            }

            a_softmax_scale_online(&tmp_o2[0], osum);
            a_cpy_tpp(&tmp_o2[0], &weighted_avg_a[i][j1][n][0]);
          }
        }
      }
    }
  }
}

lda = HS_t;
ldb = N_t * H_t;
ldc = N_t * H_t;

auto g_brgemm_tpp = SCOPEITGEMM(
    (BrgemmTPP<
        T,
        T>(C_BLOCKSIZE, N_t* H_t, HS_t, 0, 0, lda, ldb, ldc, 0.0, 0, 1)));
auto g_sigmoid_tpp =
    SCOPEIT(SiLUFwdTPP<T>(C_BLOCKSIZE, N_t* H_t, ldc, ldc), EW_MUL);
auto g_mul_tpp = SCOPEIT((MulTPP<T, T>(C_BLOCKSIZE * N_t * H_t)), EW_MUL);

// gate_values = at::sigmoid(at::add(at::einsum("bqc,chv->bqhv", {q_data,
// gating_w}), gating_b));   /* [512, 764, 8, 32]  = [512, 764, 256] * [256, 8,
// 32] + [8, 32]*/ weighted_avg = at::mul(weighted_avg, gate_values); /* [512,
// 764, 8, 32]  = [512, 764, 8, 32] * [512, 764, 8, 32] */ output =
// at::add(at::einsum("bqhc,hco->bqo", {weighted_avg, output_w}), output_b); /*
// [512, 764, 256]  = [512, 764, 8, 32] * [8, 32, 256] + [256] */

{
  RECORD_SCOPE(alpha_o_gemm, {weighted_avg, v, q_data, gating_w});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
    for (int64_t i = 0; i < B_t; i++) {
      for (int64_t j = 0; j < S_t; j += C_BLOCKSIZE) {
        T tmp[C_BLOCKSIZE * N_t * H_t];
        T tmp_gate_values[C_BLOCKSIZE * N_t * H_t];

        g_brgemm_tpp(&q_data_a[i][j][0], &gating_w_a[0][0][0], &tmp[0], 1);

        g_sigmoid_tpp(&tmp[0], &tmp[0], &tmp_gate_values[0]);
        g_mul_tpp(
            &tmp_gate_values[0],
            &weighted_avg_a[i][j][0][0],
            &output_a[i][j][0][0]);
      }
    }
  }
}

// if (S_t != Sp_t) {
//   output = output.narrow(1, 0, Sp_t);
// }

return output;

// int64_t64_t b_t = q_data.size(0);                    /* Batch (512) */
// int64_t64_t q_t = q_data.size(1);                    /* Query (764) */
// int64_t64_t k_t = m_data.size(1);                    /* Key (764) */
// int64_t64_t a_t = q_data.size(2);                  /* Channels (256) */

// int64_t64_t h_t = query_w.size(1);                  /* number of heads (8) */
// int64_t64_t c_t = query_w.size(2);                  /* head channels (32) */

// auto output = q_data.new_empty({b_t,q_t,a_t});

// auto q = q_data.new_empty({b_t,q_t,h_t,c_t});
// auto k = q_data.new_empty({b_t,k_t,h_t,c_t});
// auto v = q_data.new_empty({b_t,k_t,h_t,c_t});

// auto logits = q_data.new_empty({b_t,h_t,q_t,k_t});
// auto weights = q_data.new_empty({b_t,h_t,q_t,k_t});
// auto weighted_avg = q_data.new_empty({b_t,q_t,h_t,c_t});

// auto gate_values = q_data.new_empty({b_t,q_t,h_t,value_dim});

// q = at::mul(at::einsum("bqa,ahc->bqhc", {q_data, query_w}),
// (1.0/sqrt(key_dim))) ; k = at::einsum("bka,ahc->bkhc", {m_data, key_w}); v =
// at::einsum("bka,ahc->bkhc", {m_data, value_w});

// logits = at::add(at::einsum("bqhc,bkhc->bhqk", {q, k}), bias);

// if (nonbatched_bias.size(0) > 0)
//     logits = at::add(logits, at::unsqueeze(nonbatched_bias, 0));

// weights = at::_softmax(logits, -1, false);

// weighted_avg = at::einsum("bhqk,bkhc->bqhc", {weights, v});

// gate_values = at::sigmoid(at::add(at::einsum("bqc,chv->bqhv", {q_data,
// gating_w}), gating_b));

// weighted_avg = at::mul(weighted_avg, gate_values);

// output = at::add(at::einsum("bqhc,hco->bqo", {weighted_avg, output_w}),
// output_b);

// return output;
