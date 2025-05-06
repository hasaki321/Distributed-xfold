# Copyright 2024 xfold authors
# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md


from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from xfold.nn import atom_layout
from xfold import fastnn
from xfold.nn.diffusion_transformer import (
    AdaptiveLayerNorm, 
    AdaLNZero, 
    DiffusionTransition, 
    SelfAttention,
    DiffusionCrossAttTransformer,
    CrossAttention,
    DiffusionTransformer
)
from distributed_xfold.xsmm_kernels.prototypes.Batched_DiffusionSelfAttention import BatchedSelfAttentionXSMM_forward
from distributed_xfold.xsmm_kernels.prototypes.Batched_DiffusionCrossAttention import BatchedCrossAttentionXSMM_forward

import torch.distributed as dist
from distributed_xfold.distribute_utils import shard_linear

class AdaptiveLayerNorm(nn.Module):
    def __init__(self,
                 c_x: int,
                 c_single_cond: int,
                 use_single_cond: bool = False) -> None:

        super(AdaptiveLayerNorm, self).__init__()

        self.c_x = c_x
        self.c_single_cond = c_single_cond
        self.use_single_cond = use_single_cond

        if self.use_single_cond is True:
            self.layer_norm = fastnn.LayerNorm(
                self.c_x, elementwise_affine=False, bias=False)
            self.single_cond_layer_norm = fastnn.LayerNorm(
                self.c_single_cond, bias=False)
            self.single_cond_scale = nn.Linear(
                self.c_single_cond, self.c_x, bias=True)
            self.single_cond_bias = nn.Linear(
                self.c_single_cond, self.c_x, bias=False)
        else:
            self.layer_norm = fastnn.LayerNorm(self.c_x)

    def forward(self,
                x: torch.Tensor,
                single_cond: Optional[torch.Tensor] = None) -> torch.Tensor:

        assert (single_cond is None) == (self.use_single_cond is False)

        if self.use_single_cond is True:
            x = self.layer_norm(x)
            single_cond = self.single_cond_layer_norm(single_cond)
            single_scale = self.single_cond_scale(single_cond)
            single_bias = self.single_cond_bias(single_cond)
            return torch.sigmoid(single_scale) * x + single_bias
        else:
            return self.layer_norm(x)

# DP [y], TP[n]
class DistributeAdaptiveLayerNorm(AdaptiveLayerNorm): pass

# DP [y], TP[y]
class DistributeAdaLNZero(AdaLNZero):
    def __init__(self,
                 device_mesh,
                 c_in: int,
                 c_out: int,
                 c_single_cond: int,
                 use_single_cond: bool = False,
                 shard_out:bool = False) -> None:
        super().__init__(c_in,c_out,c_single_cond,use_single_cond)
        self.shard_out = shard_out
        self.rank = dist.get_rank()
        self.device_mesh = device_mesh
        self.use_tp = device_mesh.tp_size > 1
        self.tp_size = self.device_mesh.tp_size
        self.tp_shard_num = self.device_mesh.get_tp_shard_num(self.rank)
        self.tp_shard_group = self.device_mesh.get_tp_group(self.rank)

        self.tpp_transition2 = None 

    def shard_params(self): 
        self.transition2 = shard_linear(self.transition2, self.rank, self.device_mesh, self.shard_out)
        # if self.use_single_cond is True:
        #     self.adaptive_zero_cond = shard_linear(self.adaptive_zero_cond, self.rank, self.device_mesh, self.shard_out)

    def forward(self,
                x: torch.Tensor,
                single_cond: Optional[torch.Tensor] = None) -> torch.Tensor:

        assert (single_cond is None) == (self.use_single_cond is False)

        if self.tpp_transition2 is not None:
            output = torch.einsum('...hk, hkc -> ...c', x, self.tpp_transition2)
        else:
            output = self.transition2(x)

        if self.use_tp:
            if not self.shard_out:
                dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.tp_shard_group)

        if self.use_single_cond is True:
            cond = self.adaptive_zero_cond(single_cond)
            output = torch.sigmoid(cond) * output

        return output

class DistributeDiffusionTransition(DiffusionTransition):
    def __init__(self,
                 device_mesh,
                 c_x: int,
                 c_single_cond: int,
                 num_intermediate_factor: int = 2,
                 use_single_cond: bool = False) -> None:
        super(DistributeDiffusionTransition, self).__init__(c_x, c_single_cond, num_intermediate_factor, use_single_cond)

        self.adaptive_zero_init = DistributeAdaLNZero(
            device_mesh,
            self.num_intermediate_factor * self.c_x,
            self.c_x,
            self.c_single_cond,
            self.use_single_cond,
            shard_out = False
        )
        self.rank = dist.get_rank()
        self.device_mesh = device_mesh

    def shard_params(self): 
        self.transition1 = shard_linear(self.transition1, self.rank, self.device_mesh, True)

    def forward(self, x: torch.Tensor, single_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.adaptive_layernorm(x, single_cond)
        c = fastnn.gated_linear_unit(x, self.transition1.weight.T)
        return self.adaptive_zero_init(c, single_cond)


class DistributeSelfAttention(SelfAttention):
    def __init__(self,
                 device_mesh,
                 c_x: int = 768,
                 c_single_cond: int = 384,
                 num_head: int = 16,
                 use_single_cond: bool = False,
                 use_batch_infer: bool = False) -> None:

        super().__init__(c_x, c_single_cond, num_head, use_single_cond)

        self.adaptive_zero_init = DistributeAdaLNZero(
            device_mesh, self.c_x, self.c_x, self.c_single_cond, self.use_single_cond, shard_out=False)
        
        self.use_batch_infer = use_batch_infer

        self.rank = dist.get_rank()
        self.device_mesh = device_mesh
        self.use_tp = device_mesh.tp_size > 1
        self.tp_size = self.device_mesh.tp_size
        self.tp_shard_num = self.device_mesh.get_tp_shard_num(self.rank)
        self.tp_shard_group = self.device_mesh.get_tp_group(self.rank)

    def shard_params(self): 
        self.q_projection = shard_linear(self.q_projection, self.rank, self.device_mesh, True)
        self.k_projection = shard_linear(self.k_projection, self.rank, self.device_mesh, True)
        self.v_projection = shard_linear(self.v_projection, self.rank, self.device_mesh, True)
        self.gating_query = shard_linear(self.gating_query, self.rank, self.device_mesh, True)

    def load_xsmm_params(self): 
        dtype = torch.bfloat16
        qkv_shape = (self.c_x,) + (self.num_head // self.tp_size, self.qkv_dim)

        self.query_w = nn.Parameter(self.q_projection.weight.T.reshape(qkv_shape)\
            .to(dtype).contiguous(), requires_grad=False)
        self.query_b = nn.Parameter(self.q_projection.bias.reshape(qkv_shape[1:])\
            .to(torch.float32).contiguous(), requires_grad=False)

        self.key_w = nn.Parameter(self.k_projection.weight.T.reshape(qkv_shape)\
            .to(dtype).contiguous(), requires_grad=False)
        self.value_w = nn.Parameter(self.v_projection.weight.T.reshape(qkv_shape)\
            .to(dtype).contiguous(), requires_grad=False)

        self.gating_w = nn.Parameter(self.gating_query.weight.T.reshape(qkv_shape)\
            .to(dtype).contiguous(), requires_grad=False
        )

        self.adaptive_zero_init.tpp_transition2 = nn.Parameter(self.adaptive_zero_init.transition2.weight.T.\
            reshape((self.num_head // self.tp_size, self.qkv_dim, self.c_x)).contiguous(), 
            requires_grad=False
        )

        self.key_dim = self.value_dim = self.qkv_dim

    def xsmm_forward(self,
                x: torch.Tensor,
                mask: torch.Tensor,
                pair_logits: Optional[torch.Tensor] = None,
                single_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert (single_cond is None) == (self.use_single_cond is False)

        x = self.adaptive_layernorm(x, single_cond)

        bias = 1e9 * (mask[None, None, None, :].float() - 1)
        dtype = x.dtype

        if self.use_batch_infer:
            bias = torch.tile(bias, (x.shape[0], 1, 1, 1))
            weighted_avg = BatchedSelfAttentionXSMM_forward(self, x.bfloat16(), bias, pair_logits)
        else:
            x = x.unsqueeze(0)
            weighted_avg = BatchedSelfAttentionXSMM_forward(self, x.bfloat16(), bias, pair_logits).squeeze(0)

        weighted_avg = weighted_avg.to(dtype) # Cast back to original dtype

        return self.adaptive_zero_init(weighted_avg, single_cond)


    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor,
                pair_logits: Optional[torch.Tensor] = None,
                single_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (num_tokens, ch)
            mask (torch.Tensor): (num_tokens,)
            pair_logits (torch.Tensor, optional): (num_heads, num_tokens, num_tokens)
        """

        assert (single_cond is None) == (self.use_single_cond is False)

        x = self.adaptive_layernorm(x, single_cond)
        # assert ~torch.isnan(x).any()

        q = self.q_projection(x)
        k = self.k_projection(x)
        v = self.v_projection(x)

        if self.use_batch_infer:
            q, k, v = map(lambda t: einops.rearrange(
                t, 'b n (h c) -> b h n c', h=self.num_head // self.tp_size), [q, k, v])
        else:
            q, k, v = map(lambda t: einops.rearrange(
                t, 'n (h c) -> h n c', h=self.num_head // self.tp_size).unsqueeze(0), [q, k, v])
        # assert ~torch.isnan(q).any() and ~torch.isnan(k).any() and ~torch.isnan(v).any()

        weighted_avg = fastnn.dot_product_attention(
            q, k, v, mask=mask, bias=pair_logits
        )
        # assert ~torch.isnan(weighted_avg).any()

        if self.use_batch_infer:
            weighted_avg = einops.rearrange(weighted_avg, 'b h q c -> b q (h c)')
        else:
            weighted_avg = weighted_avg.squeeze(0)
            weighted_avg = einops.rearrange(weighted_avg, 'h q c -> q (h c)')

        gate_logits = self.gating_query(x)
        weighted_avg *= torch.sigmoid(gate_logits)
        # assert ~torch.isnan(weighted_avg).any()

        return self.adaptive_zero_init(weighted_avg, single_cond)

class DistributeDiffusionTransformer(DiffusionTransformer):
    def __init__(self,
                 device_mesh,
                 c_act: int = 768,
                 c_single_cond: int = 384,
                 c_pair_cond: int = 128,
                 num_head: int = 16,
                 num_blocks: int = 24,
                 super_block_size: int = 4,
                 use_batch_infer: bool = False) -> None:

        super(DistributeDiffusionTransformer, self).__init__(c_act, c_single_cond, c_pair_cond, num_head, num_blocks, super_block_size)
        self.pair_logits_projection = nn.ModuleList(
            [nn.Linear(self.c_pair_cond, self.super_block_size * self.num_head, bias=False) for _ in range(self.num_super_blocks)])

        self.self_attention = nn.ModuleList(
            [DistributeSelfAttention(device_mesh, self.c_act, self.c_single_cond, use_single_cond=True, use_batch_infer=use_batch_infer) for _ in range(self.num_blocks)])
        self.transition_block = nn.ModuleList(
            [DistributeDiffusionTransition(device_mesh, self.c_act, self.c_single_cond, use_single_cond=True) for _ in range(self.num_blocks)])
        
        # =======================================
        self.rank = dist.get_rank()
        self.device_mesh = device_mesh
        self.use_tp = device_mesh.tp_size > 1
        self.tp_size = self.device_mesh.tp_size

    def shard_params(self): 
        self.pair_logits_projection = nn.ModuleList([shard_linear(projection, self.rank, self.device_mesh, True) \
                                                     for projection in self.pair_logits_projection])

    def forward(self,
                act: torch.Tensor,
                mask: torch.Tensor,
                single_cond: torch.Tensor,
                pair_cond:  torch.Tensor):
        pair_act = self.pair_input_layer_norm(pair_cond)

        for super_block_i in range(self.num_super_blocks):
            pair_logits = self.pair_logits_projection[super_block_i](pair_act)

            pair_logits = einops.rearrange(
                pair_logits, 'n s (b h) -> b h n s', h=self.num_head // self.tp_size)
            for j in range(self.super_block_size):
                act += self.self_attention[super_block_i * self.super_block_size + j](act, mask, pair_logits[j, ...], single_cond)
                act += self.transition_block[super_block_i * self.super_block_size + j](act, single_cond)

        return act


class DistrubuteCrossAttention(CrossAttention):
    def __init__(self, device_mesh, 
                        key_dim: int = 128, 
                        value_dim: int = 128, 
                        c_single_cond: int = 128, 
                        num_head: int = 4, 
                        use_batch_infer:bool=False) -> None:
        super().__init__(
            key_dim,
            value_dim,
            c_single_cond,
            num_head
        )
        self.use_batch_infer = use_batch_infer
        self.adaptive_zero_init = DistributeAdaLNZero(
            device_mesh, self.value_dim, self.value_dim, self.key_dim, use_single_cond=True, shard_out=False)

        self.rank = dist.get_rank()
        self.device_mesh = device_mesh
        self.use_tp = device_mesh.tp_size > 1
        self.tp_size = self.device_mesh.tp_size

    def shard_params(self): 
        self.q_projection = shard_linear(self.q_projection, self.rank, self.device_mesh, True)
        self.k_projection = shard_linear(self.k_projection, self.rank, self.device_mesh, True)
        self.v_projection = shard_linear(self.v_projection, self.rank, self.device_mesh, True)
        self.gating_query = shard_linear(self.gating_query, self.rank, self.device_mesh, True)

    def load_xsmm_params(self): 
        dtype = torch.float32
        qk_shape = (self.key_dim,) + (self.num_head // self.tp_size, self.key_dim_per_head)
        v_shape = (self.value_dim,) + (self.num_head // self.tp_size, self.value_dim_per_head)

        self.query_w = nn.Parameter(self.q_projection.weight.T.reshape(qk_shape)\
            .to(dtype).contiguous(), requires_grad=False)
        self.query_b = nn.Parameter(self.q_projection.bias.reshape(qk_shape[1:])\
            .to(dtype).contiguous(), requires_grad=False)

        self.key_w = nn.Parameter(self.k_projection.weight.T.reshape(qk_shape)\
            .to(dtype).contiguous(), requires_grad=False)
        self.value_w = nn.Parameter(self.v_projection.weight.T.reshape(v_shape)\
            .to(dtype).contiguous(), requires_grad=False)

        self.gating_w = nn.Parameter(self.gating_query.weight.T.reshape(v_shape)\
            .to(dtype).contiguous(), requires_grad=False
        )

        self.adaptive_zero_init.tpp_transition2 = nn.Parameter(self.adaptive_zero_init.transition2.weight.T.\
            reshape((self.num_head // self.tp_size, self.value_dim_per_head, self.value_dim)).to(dtype).contiguous(), requires_grad=False
        )

        self.key_dim = self.key_dim_per_head
        self.value_dim = self.value_dim_per_head

    def _attention(self, x_q, x_k, bias):
        q = self.q_projection(x_q)                                              # ((batch),  num_subsets, num_queries,  head // tp_size, ch)
        k = self.k_projection(x_k)                                              # ((batch),  num_subsets, num_keys,     head // tp_size, ch)
        q = torch.reshape(q, q.shape[:-1] + (self.num_head // self.tp_size, self.key_dim_per_head))
        k = torch.reshape(k, k.shape[:-1] + (self.num_head // self.tp_size, self.key_dim_per_head))

        logits = torch.einsum('...qhc,...khc->...hqk',
                              q * self.q_scale, k) + bias
        weights = torch.softmax(logits, axis=-1)                                # (..., num_head // tp_size, Q, K)

        v = self.v_projection(x_k)
        v = torch.reshape(v, v.shape[:-1] +
                          (self.num_head // self.tp_size, self.value_dim_per_head))
        weighted_avg = torch.einsum('...hqk,...khc->...qhc', weights, v)        # (..., Q, num_head // tp_size, value_dim)
        weighted_avg = torch.reshape(
            weighted_avg, weighted_avg.shape[:-2] + (-1,))                      # (..., Q, num_head * value_dim // tp_size)

        gate_logits = self.gating_query(x_q)                                    # (..., Q, num_head * value_dim // tp_size)
        weighted_avg *= torch.sigmoid(gate_logits)                              # ((batch), num_subsets, num_queries, num_head * value_dim // tp_size)
        return weighted_avg

    def xsmm_forward(
        self,
        x_q: torch.Tensor,                                   # ((batch),  num_subsets, num_queries,   ch)
        x_k: torch.Tensor,                                   # ((batch),  num_subsets, num_key,       ch)
        mask_q: torch.Tensor,                                # ((1),      num_subsets, num_queries)
        mask_k: torch.Tensor,                                # ((1),      num_subsets, num_keys)
        pair_logits: torch.Tensor | None = None,             # ((1),      num_subsets, num_heads, num_queries, num_keys)
        single_cond_q: torch.Tensor | None = None,           # ((1)，     num_subsets, num_queries,   ch)
        single_cond_k: torch.Tensor | None = None,           # ((1),      num_subsets, num_keys,      ch)
    ) -> torch.Tensor:
        """Multihead self-attention."""
        # assert len(mask_q.shape) == len(x_q.shape) - 1, f'{mask_q.shape}, {x_q.shape}'
        # assert len(mask_k.shape) == len(x_k.shape) - 1, f'{mask_k.shape}, {x_k.shape}'

        # bias: ... x heads (1) x query x key
        bias = (
            1e9
            * mask_q.logical_not()[..., None, :, None]
            * mask_k.logical_not()[..., None, None, :]
        )  # ((1), num_subsets, 1, num_queries, num_keys)

        x_q = self.q_adaptive_layernorm(x_q, single_cond_q)             # ((batch),  num_subsets, num_queries,   ch)
        x_k = self.k_adaptive_layernorm(x_k, single_cond_k)             # ((batch),  num_subsets, num_keys,   ch)

        if pair_logits is not None:
            bias = bias + pair_logits
            # ((1),  num_subsets, 1, num_queries, num_keys) + ((1), num_subsets, num_heads // tp_size, num_queries, num_keys)

        if self.use_batch_infer:
            bs, ns = x_q.shape[:2]
            x_q = x_q.flatten(end_dim=1).contiguous() # (batch * num_subsets, num_queries,   ch)
            x_k = x_k.flatten(end_dim=1).contiguous() # (batch * num_subsets, num_keys,   ch)
            bias = torch.tile(bias,(bs,1,1,1,1)).flatten(end_dim=1).contiguous() # (batch * num_subsets, num_keys,   ch)

        dtype = x_q.dtype

        # (batch * num_subsets, num_heads, num_queries, ch)
        weighted_avg = BatchedCrossAttentionXSMM_forward(self, q_data=x_q, m_data=x_k, batched_bias=bias)

        weighted_avg = weighted_avg.to(dtype) # Cast back to original dtype

        if self.use_batch_infer:
            weighted_avg = weighted_avg.reshape((bs, ns) + weighted_avg.shape[1:]) # (batch, num_subsets, num_heads, num_queries, ch)

        # Adaptive zero initialization
        output = self.adaptive_zero_init(weighted_avg, single_cond_q)   # (1，num_subsets, num_queries, ch)
        return output      

    def forward(
        self,
        x_q: torch.Tensor,                                   # ((batch),  num_subsets, num_queries,   ch)
        x_k: torch.Tensor,                                   # ((batch),  num_subsets, num_key,       ch)
        mask_q: torch.Tensor,                                # ((1),      num_subsets, num_queries)
        mask_k: torch.Tensor,                                # ((1),      num_subsets, num_keys)
        pair_logits: torch.Tensor | None = None,             # ((1),      num_subsets, num_heads, num_queries, num_keys)
        single_cond_q: torch.Tensor | None = None,           # ((1)，     num_subsets, num_queries,   ch)
        single_cond_k: torch.Tensor | None = None,           # ((1),      num_subsets, num_keys,      ch)
    ) -> torch.Tensor:
        """Multihead self-attention."""
        # assert len(mask_q.shape) == len(x_q.shape) - 1, f'{mask_q.shape}, {x_q.shape}'
        # assert len(mask_k.shape) == len(x_k.shape) - 1, f'{mask_k.shape}, {x_k.shape}'

        # bias: ... x heads (1) x query x key
        bias = (
            1e9
            * mask_q.logical_not()[..., None, :, None]
            * mask_k.logical_not()[..., None, None, :]
        )  # ((1), num_subsets, 1, num_queries, num_keys)

        x_q = self.q_adaptive_layernorm(x_q, single_cond_q)             # ((batch),  num_subsets, num_queries,   ch)
        x_k = self.k_adaptive_layernorm(x_k, single_cond_k)             # ((batch),  num_subsets, num_keys,   ch)
        if pair_logits is not None:
            bias = bias + pair_logits
            # ((1),  num_subsets, 1, num_queries, num_keys) + ((1), num_subsets, num_heads // tp_size, num_queries, num_keys)

        weighted_avg = self._attention(x_q=x_q, x_k=x_k, bias=bias)
        
        # Adaptive zero initialization
        output = self.adaptive_zero_init(weighted_avg, single_cond_q)   # (1，num_subsets, num_queries, ch)
        return output                                                   # (batch，num_subsets, num_queries, ch)

class DistributeDiffusionCrossAttTransformer(DiffusionCrossAttTransformer):
    """Transformer with cross attention between two sets of subsets in PyTorch."""
    def __init__(self, device_mesh, 
                 c_query: int = 128, 
                 c_single_cond: int = 128, 
                 c_pair_cond: int = 16, 
                 num_blocks: int = 3, 
                 num_head: int = 4, 
                 use_batch_infer:bool=True) -> None:
        super().__init__(
            c_query,
            c_single_cond,
            c_pair_cond,
            num_blocks,
            num_head
        )
        self.use_batch_infer = use_batch_infer

        self.cross_attention = nn.ModuleList(
            [DistrubuteCrossAttention(device_mesh, num_head=self.num_head, use_batch_infer=use_batch_infer) for _ in range(self.num_blocks)])

        self.transition_block = nn.ModuleList(
            [DistributeDiffusionTransition(device_mesh, c_x=self.c_query, c_single_cond=self.c_single_cond, use_single_cond=True) for _ in range(self.num_blocks)])
    
        self.rank = dist.get_rank()
        self.device_mesh = device_mesh
        self.tp_size = self.device_mesh.tp_size
    
    def shard_params(self): 
        self.pair_logits_projection = shard_linear(self.pair_logits_projection, self.rank, self.device_mesh, True)
    
    def forward(
        self,
        queries_act: torch.Tensor,                                                  # ((batch), num_subsets, num_queries, ch)
        queries_mask: torch.Tensor,                                                 # (num_subsets, num_queries)
        queries_to_keys: atom_layout.GatherInfo,                                    # (num_subsets, num_keys)
        keys_mask: torch.Tensor,                                                    # (num_subsets, num_keys)
        queries_single_cond: torch.Tensor,                                          # (num_subsets, num_queries, ch)
        keys_single_cond: torch.Tensor,                                             # (num_subsets, num_keys, ch)
        pair_cond: torch.Tensor,                                                    # (num_subsets, num_queries, num_keys, ch)
    ) -> torch.Tensor:
        """Forward pass of CrossAttTransformer in PyTorch."""
        
        # Precompute pair logits for performance
        pair_act = self.pair_input_layer_norm(pair_cond)

        # (num_subsets, num_queries, num_keys, num_blocks, num_heads)
        pair_logits = self.pair_logits_projection(pair_act)

        # (num_block, num_subsets, num_heads, num_queries, num_keys)
        pair_logits = einops.rearrange(
            pair_logits, 'n q k (b h) -> b n h q k', h=self.num_head // self.tp_size)

        if self.use_batch_infer:
            queries_mask=queries_mask[None, ...]
            keys_mask=keys_mask[None, ...]
            pair_logits=pair_logits[:, None, ...]
            queries_single_cond=queries_single_cond[None, ...]
            keys_single_cond=keys_single_cond[None, ...]

        # Cross attention blocks
        for block_idx in range(self.num_blocks):
            keys_act = atom_layout.convert(
                queries_to_keys, queries_act, layout_axes=(-3, -2)
            )                                                                       # (batch, num_subsets, num_key, ch)
            queries_act += self.cross_attention[block_idx](
                x_q=queries_act,                                                    # ((batch), num_subsets, num_queries, ch)
                x_k=keys_act,                                                       # ((batch), num_subsets, num_key, ch)
                mask_q=queries_mask,                                                # ((1), num_subsets, num_queries)
                mask_k=keys_mask,                                                   # ((1), num_subsets, num_keys)
                pair_logits=pair_logits[block_idx],                                 # ((1), num_subsets, num_heads // tp_size, num_queries, num_keys)
                single_cond_q=queries_single_cond,                                  # ((1), num_subsets, num_queries, ch)
                single_cond_k=keys_single_cond,                                     # ((1), num_subsets, num_keys, ch)
            )
            queries_act += self.transition_block[block_idx](
                queries_act,                                                        # ((batch), num_subsets, num_queries, ch)
                queries_single_cond,                                                # ((1), num_subsets, num_queries, ch)
            )
            
        return queries_act # (batch，num_subsets, num_queries, ch)