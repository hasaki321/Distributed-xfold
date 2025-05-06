RECORD_FUNCTION(
    "Transition forward",
    std::vector<c10::IValue>({act})); // For recording time

int64_t B_t = act.size(0);
int64_t Sp_t = act.size(1);
int64_t act_dim = act.size(2);

int64_t num_intermediate_channel = transition1.size(1) / 2;

int64_t S_t = Sp_t;
if (Sp_t % T_BLOCKSIZE != 0) {
  S_t = (Sp_t / T_BLOCKSIZE + 1) * T_BLOCKSIZE;
  auto act_pad = act.new_zeros({B_t, S_t - Sp_t, act_dim});
  act = at::cat({act, act_pad}, 1);
}

auto act_a = GetVLAPtr<T>(act, {S_t, act_dim});

auto transition1_a = GetVLAPtr<T>(transition1, {num_intermediate_channel * 2});
auto transition2_a = GetVLAPtr<T>(transition2, {act_dim});

auto output = act.new_empty({B_t, S_t, act_dim}); /* [512, 764, 256] */
auto output_a = GetVLAPtr<T>(output, {S_t, act_dim});

auto layernorm = SCOPEIT(LayerNormFwdTPP<T>(1, T_BLOCKSIZE, act_dim, 0.00001), LAYER_NORM);
auto input_gamma_a = GetVLAPtr<T>(layernorm_weight, {1L});
auto input_beta_a = GetVLAPtr<T>(layernorm_bias, {1L});

#pragma omp parallel for collapse(2)
for (int64_t i = 0; i < B_t; i++) {
  for (int64_t j = 0; j < S_t; j += T_BLOCKSIZE) {
    T tmp_mean[act_dim];
    T tmp_var[act_dim];
    layernorm(
        &act_a[i][j][0],
        &input_gamma_a[0][0],
        &input_beta_a[0][0],
        &tmp_mean[0],
        &tmp_var[0],
        &act_a[i][j][0]);
  }
}
// printf("ACT shape %d", 1);

// act = at::layer_norm(act, act_dim, layernorm_weight,
// layernorm_bias).contiguous();
act_a = GetVLAPtr<T>(act, {S_t, act_dim});

auto t1_brgemm_tpp = SCOPEITGEMM(
    (BrgemmTPP<T,T>(T_BLOCKSIZE, act_dim, num_intermediate_channel * 2, 0, 0, act_dim, num_intermediate_channel * 2, num_intermediate_channel * 2, 0.0, 0, 1))
);
auto proj_copy_tpp = SCOPEIT(CpyTPP<T>(T_BLOCKSIZE, num_intermediate_channel, num_intermediate_channel * 2, num_intermediate_channel), EW_COPY);
auto g_silu_tpp = SCOPEIT(SiLUFwdTPP<T>(T_BLOCKSIZE, num_intermediate_channel, num_intermediate_channel * 2, num_intermediate_channel), EW_MUL);
auto g_mul_tpp = SCOPEIT((MulTPP<T, T>(T_BLOCKSIZE * num_intermediate_channel)), EW_MUL);

auto t2_brgemm_tpp = SCOPEITGEMM(
    (BrgemmTPP<T,T>(T_BLOCKSIZE, num_intermediate_channel, act_dim, 0, 0, num_intermediate_channel, act_dim, act_dim, 0.0, 0, 1))
);

{
  RECORD_SCOPE(alpha_t_gemm, {act, transition1, transition2});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
    #pragma omp parallel for collapse(2)
    for (int64_t i = 0; i < B_t; i++) {
      for (int64_t j = 0; j < S_t; j += T_BLOCKSIZE) {
        T tmp[T_BLOCKSIZE][num_intermediate_channel * 2];
        T tmp_proj_values[T_BLOCKSIZE * num_intermediate_channel];
        T tmp_gate_values[T_BLOCKSIZE * num_intermediate_channel];
        T tmp_gate_outputs[T_BLOCKSIZE * num_intermediate_channel];
        t1_brgemm_tpp(&act_a[i][j][0], &transition1_a[0][0], &tmp[0][0], 1);
        // printf("Tmp shape %d", 1);

        g_silu_tpp(&tmp[0][0], &tmp_gate_values[0]);
        proj_copy_tpp(&tmp[0][num_intermediate_channel], &tmp_proj_values[0]);
        // printf("Gate shape %d", 1);

        g_mul_tpp(&tmp_gate_values[0],
                  &tmp_proj_values[0],
                  &tmp_gate_outputs[0]);
        // printf("Gate out shape %d", 1);

        t2_brgemm_tpp(&tmp_gate_outputs[0], &transition2_a[0][0], &output_a[i][j][0], 1);

      }
    }
  }
}
// printf("out shape %d", 1);
if (S_t != Sp_t) {
  output = output.narrow(1, 0, Sp_t);
}

return output;