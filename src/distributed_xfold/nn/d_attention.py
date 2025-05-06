# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md


import einops
import torch
import torch.nn as nn

from xfold import fastnn
from xfold.nn.attention import MSAAttention, GridSelfAttention
from distributed_xfold.xsmm_kernels.prototypes.GridSelfAttention import GridSelfAttentionXSMM_forward
from distributed_xfold.distribute_utils import all_gather_into_tensor, ShardInfo, shard_linear
import torch.distributed as dist


class DistributeGridSelfAttention(GridSelfAttention):
    def __init__(self, device_mesh, c_pair: int = 128, num_head: int = 4, transpose: bool = False):
        super().__init__(c_pair, num_head, transpose)
        self.rank = dist.get_rank()
        self.device_mesh = device_mesh
        self.use_tp = device_mesh.tp_size > 1
        self.use_dp = device_mesh.dp_size > 1
        self.dp_size = self.device_mesh.dp_size
        self.tp_size = self.device_mesh.tp_size
        self.dp_shard_num = self.device_mesh.get_dp_shard_num(self.rank)
        self.tp_shard_num = self.device_mesh.get_tp_shard_num(self.rank)
        self.dp_shard_group = self.device_mesh.get_dp_group(self.rank)
        self.tp_shard_group = self.device_mesh.get_tp_group(self.rank)

    def shard_params(self): 
        self.q_projection = shard_linear(self.q_projection, self.rank, self.device_mesh, True)
        self.k_projection = shard_linear(self.k_projection, self.rank, self.device_mesh, True)
        self.v_projection = shard_linear(self.v_projection, self.rank, self.device_mesh, True)
        self.gating_query = shard_linear(self.gating_query, self.rank, self.device_mesh, True)

        self.output_projection = shard_linear(self.output_projection, self.rank, self.device_mesh, False)

    def load_xsmm_params(self): 
        dtype = torch.bfloat16
        qkv_shape = (self.c_pair,) + (self.num_head // self.tp_size, self.qkv_dim)

        self.query_w = nn.Parameter(self.q_projection.weight.T.reshape(qkv_shape)\
            .to(dtype).contiguous(), requires_grad=False)
        self.key_w = nn.Parameter(self.k_projection.weight.T.reshape(qkv_shape)\
            .to(dtype).contiguous(), requires_grad=False)
        self.value_w = nn.Parameter(self.v_projection.weight.T.reshape(qkv_shape)\
            .to(dtype).contiguous(), requires_grad=False)

        self.gating_w = nn.Parameter(self.gating_query.weight.T.reshape(qkv_shape)\
            .to(dtype).contiguous(), requires_grad=False
        )
        self.output_w = nn.Parameter(
            self.output_projection.weight.T.reshape(self.num_head // self.tp_size, self.qkv_dim, self.c_pair)\
                .to(dtype).contiguous(), requires_grad=False
        )

        self.key_dim = self.value_dim = self.qkv_dim


    def xsmm_forward(self, pair, mask, full_pair=None):
        if full_pair is None: full_pair = pair

        pair = self.act_norm(pair)
        nonbatched_bias = self.pair_bias_projection(self.act_norm(full_pair)).permute(2, 0, 1)
        bias = 1e9 * (mask[:, None, None, :] - 1)

        if self.transpose:
            pair = pair.permute(1, 0, 2)

        dtype = pair.dtype
        # is_16bit = dtype in [torch.float16, torch.bfloat16] # Correct dtype check for PyTorch
        # if is_16bit:
        #     pair = pair.float().contiguous() # Upcast to float32
        #     bias = bias.float().contiguous() # Upcast to float32
        #     nonbatched_bias = nonbatched_bias.float().contiguous() # Upcast to float32
        # print(pair.shape, bias.shape, nonbatched_bias.shape)
        pair = GridSelfAttentionXSMM_forward(self, pair.bfloat16(), bias, nonbatched_bias)

        # if is_16bit:
        pair = pair.to(dtype) # Cast back to original dtype
        
        if self.use_tp:
            dist.all_reduce(pair, op=dist.ReduceOp.SUM, group=self.tp_shard_group)
            
        if self.transpose:
            pair = pair.permute(1, 0, 2)
        return pair


    def forward(self, pair, mask, full_pair=None):
        """
        Args:
            pair (torch.Tensor): [N_token // dp_size, N_token, c_pair]
            mask (torch.Tensor): [N_token // dp_size, N_token]
        Returns:
            torch.Tensor: [N_token, N_token, c_pair]
        """
        if full_pair is None: full_pair = pair

        pair = self.act_norm(pair)
        nonbatched_bias = self.pair_bias_projection(self.act_norm(full_pair)).permute(2, 0, 1)

        if self.transpose:
            pair = pair.permute(1, 0, 2)
        pair = self._attention(pair, mask, nonbatched_bias)
        
        if self.use_tp:
            dist.all_reduce(pair, op=dist.ReduceOp.SUM, group=self.tp_shard_group)
            
        if self.transpose:
            pair = pair.permute(1, 0, 2)

        return pair


class DistributeMSAAttention(MSAAttention):
    def __init__(self, device_mesh, c_msa=64, c_pair=128, num_head=8):
        super().__init__(c_msa, c_pair, num_head)
        self.rank = dist.get_rank()
        self.device_mesh = device_mesh
        self.use_tp = device_mesh.tp_size > 1
        self.use_dp = device_mesh.dp_size > 1
        self.dp_size = self.device_mesh.dp_size
        self.tp_size = self.device_mesh.tp_size
        self.dp_shard_num = self.device_mesh.get_dp_shard_num(self.rank)
        self.tp_shard_num = self.device_mesh.get_tp_shard_num(self.rank)
        self.dp_shard_group = self.device_mesh.get_dp_group(self.rank)
        self.tp_shard_group = self.device_mesh.get_tp_group(self.rank)

    def shard_params(self): 
        self.pair_logits = shard_linear(self.pair_logits, self.rank, self.device_mesh, True)
        self.v_projection = shard_linear(self.v_projection, self.rank, self.device_mesh, True)
        self.gating_query = shard_linear(self.gating_query, self.rank, self.device_mesh, True)
        self.output_projection = shard_linear(self.output_projection, self.rank, self.device_mesh, False)

    def forward(self, msa, msa_mask, pair):
        batch, n_token, c = msa.shape
        msa = self.act_norm(msa)
        pair = self.pair_norm(pair)
        logits = self.pair_logits(pair)                                     
        logits = logits.permute(2, 0, 1)                                    # [num_head // tp_size, N_token, N_token]

        logits += 1e9 * (torch.max(msa_mask, dim=0).values - 1.0)
        weights = torch.softmax(logits, dim=-1)                             # [num_head // tp_size, N_token // dp_size, N_token]

        v = self.v_projection(msa)                                          # [batch, N_token, (num_head value_dim) // tp_size]
        v = einops.rearrange(v, 'b k (h c) -> b k h c', h=self.num_head // self.tp_size)    # [batch, N_token, num_head // tp_size, value_dim]

        v_avg = torch.einsum('hqk, bkhc -> bqhc', weights, v)               # [batch, N_token // dp_size, num_head // tp_size, value_dim]
        v_avg = torch.reshape(v_avg, v_avg.shape[:-2] + (-1,))

        # if self.use_dp:
        #     msa = torch.chunk(msa, self.dp_size, dim=1)[self.dp_shard_num]
        gate_values = self.gating_query(msa)                                # [batch, N_token, c_msa // tp_size]
        v_avg *= torch.sigmoid(gate_values)                                 # [batch, N_token, c_msa // tp_size]
        out = self.output_projection(v_avg)                                 # [batch, N_token, c_msa]

        if self.use_tp:
            dist.all_reduce(out.contiguous(), op=dist.ReduceOp.SUM, group=self.tp_shard_group, async_op=False)   # [b, d // dp_size, f]

        # if self.use_dp:
        #     unsharded_out = torch.zeros((batch, n_token, c), dtype=out.dtype, device=out.device)
        #     all_gather_into_tensor(unsharded_out, out, 
        #                         group=self.dp_shard_group, 
        #                         async_op=False,)
        #     out = unsharded_out
        return out
