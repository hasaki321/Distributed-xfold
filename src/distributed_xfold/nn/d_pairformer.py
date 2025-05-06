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

from xfold.nn.primitives import Transition, OuterProductMean
from xfold.nn.triangle_multiplication import TriangleMultiplication
from xfold.nn.attention import GridSelfAttention, MSAAttention
from xfold.nn.diffusion_transformer import SelfAttention
import torch.distributed as dist
from distributed_xfold.distribute_utils import all_gather_into_tensor, ShardInfo, DeviceMesh, all_to_all
from distributed_xfold.nn.d_triangle_multiplication import DistributeTriangleMultiplication, XSMMTriangleMultiplication
from distributed_xfold.nn.d_attention import DistributeGridSelfAttention, DistributeMSAAttention
from distributed_xfold.nn.d_primitives import DistributeTransition, DistributeOuterProductMean
from distributed_xfold.distribute_utils import shard_linear
from distributed_xfold.nn.d_diffusion_transformer import DistributeSelfAttention
from xfold import fastnn


class DistributePairformerBlock(nn.Module):
    """Implements Algorithm 17 [Line2-Line8] in AF3
    Ref to: openfold/model/evoformer.py and protenix/model/modules/pairformer.py
    """

    def __init__(
        self,
        device_mesh: DeviceMesh,
        n_heads: int = 16,
        c_pair: int = 128,
        c_single: int = 384,
        c_hidden_mul: int = 128,
        n_heads_pair: int = 4,
        num_intermediate_factor: int = 4,
        with_single: bool = True,
    ) -> None:
        """
        Args:
            n_heads (int, optional): number of head [for SelfAttention]. Defaults to 16.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
            c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
            c_hidden_mul (int, optional): hidden dim [for TriangleMultiplicationOutgoing].
                Defaults to 128.
            n_heads_pair (int, optional): number of head [for TriangleAttention]. Defaults to 4.
        """
        super(DistributePairformerBlock, self).__init__()
        self.rank = dist.get_rank()
        self.device_mesh = device_mesh
        self.n_heads = n_heads
        self.with_single = with_single
        self.num_intermediate_factor = num_intermediate_factor
        self.c_pair = c_pair

        # self.triangle_multiplication_outgoing = DistributeTriangleMultiplication(
        #     device_mesh, c_pair=c_pair, _outgoing=True
        # )
        # self.triangle_multiplication_incoming = DistributeTriangleMultiplication(
        #     device_mesh, c_pair=c_pair, _outgoing=False)
        
        self.triangle_multiplication_outgoing = XSMMTriangleMultiplication(
            c_pair=c_pair, _outgoing=True
        )
        self.triangle_multiplication_incoming = XSMMTriangleMultiplication(
            c_pair=c_pair, _outgoing=False)
        
        self.pair_attention1 = DistributeGridSelfAttention(
            device_mesh, c_pair=c_pair, num_head=n_heads_pair, transpose=False
        )
        self.pair_attention2 = DistributeGridSelfAttention(
            device_mesh, c_pair=c_pair, num_head=n_heads_pair, transpose=True
        )
        self.pair_transition = DistributeTransition(
            device_mesh, c_x=c_pair, num_intermediate_factor=self.num_intermediate_factor)
        self.c_single = c_single    
        if self.with_single is True:
            self.single_pair_logits_norm = fastnn.LayerNorm(c_pair)
            self.single_pair_logits_projection = nn.Linear(
                c_pair, n_heads, bias=False)
            self.single_attention_ = DistributeSelfAttention(
                device_mesh, c_x=c_single, num_head=n_heads, use_single_cond=False, use_batch_infer=False)
            self.single_transition = Transition(c_x=self.c_single)

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
        if self.with_single is True:
            self.single_pair_logits_projection = shard_linear(self.single_pair_logits_projection, self.rank, self.device_mesh, True)

    def forward(
        self,
        pair: torch.Tensor,
        pair_mask: torch.Tensor,
        single: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
        head: int = 0
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass of the PairformerBlock.

        Args:
            pair (torch.Tensor): [N_token, N_token, c_pair]
            pair_mask (torch.Tensor): [N_token, N_token]
            single (torch.Tensor, optional): [N_token, c_single]
            seq_mask (torch.Tensor, optional): [N_token]

        Returns:
            tuple[torch.Tensor, Optional[torch.Tensor]]: pair, single
        """
        n_token = max(pair.shape[:-1])
        shard_slice = ShardInfo(n_token, self.dp_size).get_shard_slice_list()[self.dp_shard_num]
    
        pair += self.triangle_multiplication_outgoing(pair, mask=pair_mask)                                 # [N_token, N_token, c_pair]
        pair += self.triangle_multiplication_incoming(pair, mask=pair_mask)                                 # [N_token, N_token, c_pair]

        if self.use_dp:
            shard_pair = pair[shard_slice]
            shard_pair_mask = pair_mask[shard_slice]
        else:
            shard_pair = pair
            shard_pair_mask = pair_mask

        shard_pair += self.pair_attention1(shard_pair, mask=shard_pair_mask, full_pair=pair)                             # [N_token // dp_size, N_token, c_pair]

        if self.use_dp:
            pair = torch.empty((n_token, n_token, self.c_pair), dtype=pair.dtype, device=pair.device)
            all_gather_into_tensor(
                    pair,
                    shard_pair,
                    group=self.dp_shard_group,
                    async_op=False
                )
            shard_pair = pair[:, shard_slice]                           # [N_token, N_token // dp_size, c_pair]

        shard_pair += self.pair_attention2(shard_pair, mask=shard_pair_mask, full_pair=pair)                           # [N_token, N_token // dp_size, c_pair]

        shard_pair += self.pair_transition(shard_pair)                                                                  # [N_token, N_token // dp_size, c_pair]

        if self.use_dp:
            pair = torch.empty((n_token, n_token, self.c_pair), dtype=pair.dtype, device=pair.device)
            all_gather_into_tensor(
                    pair,
                    shard_pair,
                    group=self.dp_shard_group,
                    async_op=False
                )
        else: pair = shard_pair

        if self.with_single is True:

            pair_logits = self.single_pair_logits_projection(self.single_pair_logits_norm(pair)) # [N_token, N_token, c_pair // tp_size]

            pair_logits = pair_logits.permute(2, 0, 1)

            single += self.single_attention_(single,
                                             seq_mask,
                                             pair_logits=pair_logits)

            single += self.single_transition(single)
            return pair, single
        else:
            return pair                                                                           # [N_token // dp_size, N_token, c_pair]


class DistributeEvoformerBlock(nn.Module):
    def __init__(self, device_mesh, c_msa: int = 64, c_pair: int = 128, n_heads_pair: int = 4) -> None:
        super(DistributeEvoformerBlock, self).__init__()
        self.c_pair = c_pair
        self.outer_product_mean = DistributeOuterProductMean(
            device_mesh, c_msa=c_msa, num_output_channel=c_pair)
        # self.outer_product_mean = OuterProductMean(
        #     c_msa=c_msa, num_output_channel=c_pair)
        self.msa_attention1 = DistributeMSAAttention(device_mesh, c_msa=c_msa, c_pair=c_pair)
        self.msa_transition = DistributeTransition(device_mesh, c_x=c_msa)

        self.triangle_multiplication_outgoing = XSMMTriangleMultiplication(
            c_pair=c_pair, _outgoing=True
        )
        self.triangle_multiplication_incoming = XSMMTriangleMultiplication(
            c_pair=c_pair, _outgoing=False)
        
        self.pair_attention1 = DistributeGridSelfAttention(
            device_mesh, c_pair=c_pair, num_head=n_heads_pair, transpose=False
        )
        self.pair_attention2 = DistributeGridSelfAttention(
            device_mesh, c_pair=c_pair, num_head=n_heads_pair, transpose=True
        )
        self.pair_transition = DistributeTransition(device_mesh, c_x=c_pair)

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

    def forward(
        self,
        msa: torch.Tensor,
        pair: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_token = max(pair.shape[:-1])
        shard_slice = ShardInfo(n_token, self.dp_size).get_shard_slice_list()[self.dp_shard_num]
        pair_outer = self.outer_product_mean(msa, msa_mask)                                                     # [N_token, N_token, c_pair]

        pair += pair_outer
        msa += self.msa_attention1(msa, msa_mask, pair)
        msa += self.msa_transition(msa)

        pair += self.triangle_multiplication_outgoing(pair, mask=pair_mask)                                     # [N_token, N_token, c_pair]
        pair += self.triangle_multiplication_incoming(pair, mask=pair_mask.T)                                   # [N_token, N_token, c_pair]

        if self.use_dp:
            shard_pair = pair[shard_slice]
            shard_pair_mask = pair_mask[shard_slice]
        else:
            shard_pair = pair
            shard_pair_mask = pair_mask

        shard_pair += self.pair_attention1(shard_pair, mask=shard_pair_mask, full_pair=pair)                             # [N_token // dp_size, N_token, c_pair]

        if self.use_dp:
            pair = torch.empty((n_token, n_token, self.c_pair), dtype=pair.dtype, device=pair.device)
            all_gather_into_tensor(
                    pair,
                    shard_pair,
                    group=self.dp_shard_group,
                    async_op=False
                )
            shard_pair = pair[:, shard_slice]                           # [N_token, N_token // dp_size, c_pair]

        shard_pair += self.pair_attention2(shard_pair, mask=shard_pair_mask, full_pair=pair)                           # [N_token, N_token // dp_size, c_pair]

        shard_pair += self.pair_transition(shard_pair)                                                                  # [N_token, N_token // dp_size, c_pair]

        if self.use_dp:
            pair = torch.empty((n_token, n_token, self.c_pair), dtype=pair.dtype, device=pair.device)
            all_gather_into_tensor(
                    pair,
                    shard_pair,
                    group=self.dp_shard_group,
                    async_op=False
                )
        else: pair = shard_pair

        return msa, pair
