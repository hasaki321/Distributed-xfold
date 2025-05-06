# Copyright 2024 distributed-xfold authors
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


import torch
import torch.nn as nn
import torch.distributed as dist

from xfold import feat_batch
from xfold.nn import featurization, utils
from xfold.nn.diffusion_transformer import DiffusionTransformer, DiffusionTransition
from xfold.nn.atom_cross_attention import AtomCrossAttEncoder, AtomCrossAttDecoder

from distributed_xfold.distribute_utils import ShardInfo, all_gather_into_tensor

from collections.abc import Callable, Sequence
from xfold.nn.diffusion_head import (
    FourierEmbeddings, 
    noise_schedule,
    random_augmentation, 
    random_rotation,
    DiffusionHead
)
from distributed_xfold.nn.d_atom_cross_attention import (
    DistributeAtomCrossAttEncoder,
    DistributeAtomCrossAttDecoder,
)
from distributed_xfold.nn.d_diffusion_transformer import (
    DistributeDiffusionTransition,
    DistributeDiffusionTransformer,
)
from xfold import fastnn

# Carefully measured by averaging multimer training set.
SIGMA_DATA = 16.0

class DistrubuteFourierEmbeddings(FourierEmbeddings):
    def __init__(self, device_mesh, dim: int):
        super(FourierEmbeddings, self).__init__()
        self.dim = dim

        self.weight = nn.Parameter(torch.ones(dim,))
        self.bias = nn.Parameter(torch.ones(dim,))

        self.rank = dist.get_rank()
        self.device_mesh = device_mesh
        self.use_tp = device_mesh.tp_size > 1
        self.tp_size = self.device_mesh.tp_size
        self.tp_shard_num = self.device_mesh.get_tp_shard_num(self.rank)

    def shard_param(self):
        self.weight = nn.Parameter(torch.chunk(self.weight, self.tp_size)[self.tp_shard_num], dtype=self.weight.dtype, device=self.weight.device)
        self.bias = nn.Parameter(torch.chunk(self.bias, self.tp_size)[self.tp_shard_num], dtype=self.bias.dtype, device=self.bias.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if not hasattr(self, "weight") or not hasattr(self, "bias"):
            raise RuntimeError("FourierEmbeddings not initialized")            

        return torch.cos(2 * torch.pi * (x[..., None] * self.weight + self.bias))

class DistributeDiffusionHead(DiffusionHead):
    def __init__(self, device_mesh, use_batch_infer):
        super().__init__()
        self.pair_transition_0 = DistributeDiffusionTransition(
            device_mesh, c_x=self.pair_channel, c_single_cond=None)
        self.pair_transition_1 = DistributeDiffusionTransition(
            device_mesh, c_x=self.pair_channel, c_single_cond=None)

        self.single_transition_0 = DistributeDiffusionTransition(
            device_mesh, c_x=self.seq_channel, c_single_cond=None)
        self.single_transition_1 = DistributeDiffusionTransition(
            device_mesh, c_x=self.seq_channel, c_single_cond=None)

        # self.fourier_embeddings = DistrubuteFourierEmbeddings(device_mesh, dim=256)

        self.atom_cross_att_encoder = DistributeAtomCrossAttEncoder(
                                                          device_mesh,  
                                                          per_token_channels=self.c_act,
                                                          with_token_atoms_act=True,
                                                          with_trunk_pair_cond=True,
                                                          with_trunk_single_cond=True,
                                                          use_batch_infer=use_batch_infer)
        self.transformer = DistributeDiffusionTransformer(device_mesh, use_batch_infer=use_batch_infer)
        self.atom_cross_att_decoder = DistributeAtomCrossAttDecoder(device_mesh, use_batch_infer=use_batch_infer)

        # ============================================================
        self.use_batch_infer = use_batch_infer
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

    def pre_conditioning(
        self,
        batch,
        embeddings: dict[str, torch.Tensor],
        noise_level: torch.Tensor,
        use_conditioning: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        single_embedding = use_conditioning * embeddings['single']
        pair_embedding = use_conditioning * embeddings['pair']

        rel_features = featurization.create_relative_encoding(
            batch.token_features, max_relative_idx=32, max_relative_chain=2
        ).to(dtype=pair_embedding.dtype)
        features_2d = torch.concatenate([pair_embedding, rel_features], dim=-1)

        target_feat = embeddings['target_feat']
        features_1d = torch.concatenate(
            [single_embedding, target_feat], dim=-1)
        
        if self.use_dp:
            b1d = features_1d.shape[0]
            b2d = features_1d.shape[0]
            f1d_shard_info = ShardInfo(b1d, self.dp_size).get_shard_slice_list()
            f2d_shard_info = ShardInfo(b2d, self.dp_size).get_shard_slice_list()
            features_1d = features_1d[f1d_shard_info[self.dp_shard_num]]             
            features_2d = features_2d[f2d_shard_info[self.dp_shard_num]]      
        
        # ==================== Pair feat ========================
        pair_cond = self.pair_cond_initial_projection(
            self.pair_cond_initial_norm(features_2d)
        )

        pair_cond += self.pair_transition_0(pair_cond)
        pair_cond += self.pair_transition_1(pair_cond)

        # ==================== Noise feat ========================
        noise_embedding = self.fourier_embeddings(
            (1 / 4) * torch.log(noise_level / SIGMA_DATA)
        )
        noise_proj = self.noise_embedding_initial_projection(
            self.noise_embedding_initial_norm(noise_embedding)
        )
        # ==================== Single feat ========================
        single_cond = self.single_cond_initial_projection(
            self.single_cond_initial_norm(features_1d))
        # print(single_cond.shape, noise_proj.shape)
        single_cond = single_cond[None, ...] + noise_proj[:, None, :]       # [1, N_token, ch] + [t, 1, ch]

        single_cond += self.single_transition_0(single_cond)
        single_cond += self.single_transition_1(single_cond)

        if self.use_dp:
            # [TODO] some problems here if using gloo backend
            unsharded_single_cond = torch.zeros((single_cond.shape[0], b1d, single_cond.shape[-1]), dtype=single_cond.dtype, device=single_cond.device)
            # print(unsharded_single_cond.shape, single_cond.shape)
            unsharded_single_cond, async_1d = all_gather_into_tensor(unsharded_single_cond, single_cond, 
                                group=self.dp_shard_group, 
                                async_op=True,)
            
            unsharded_pair_cond = torch.zeros((b2d,) + pair_cond.shape[1:], dtype=pair_cond.dtype, device=pair_cond.device)
            # print(unsharded_pair_cond.shape, pair_cond.shape)
            unsharded_pair_cond, async_2d = all_gather_into_tensor(unsharded_pair_cond, pair_cond, 
                                group=self.dp_shard_group, 
                                async_op=True,)
            async_1d.wait()
            async_2d.wait()
            backend = torch.distributed.get_backend()
            if backend == 'gloo' or backend == 'mpi':
                single_cond = torch.cat(unsharded_single_cond, dim=1)
                pair_cond = torch.cat(unsharded_pair_cond, dim=0)
            else:
                single_cond = unsharded_single_cond
                pair_cond = unsharded_pair_cond

        return single_cond, pair_cond

    def forward(
        self,
        positions_noisy: torch.Tensor,
        noise_level: torch.Tensor,
        batch: feat_batch.Batch,
        embeddings: dict[str, torch.Tensor],
        trunk_single_cond: torch.Tensor,
        trunk_pair_cond: torch.Tensor,
    ) -> torch.Tensor:
        # Extract features
        sequence_mask = batch.token_features.mask                               # [num_token]
        atom_mask = batch.predicted_structure_info.atom_mask                    # [num_token, num_atom]
        atom_mask = (atom_mask[None, ..., None] if self.use_batch_infer else atom_mask[..., None])

        # Position features 
        act = positions_noisy * atom_mask                                       # [(num_samples), num_token, num_atom, 3] * [(1), num_token, num_atom, 1]
        act = act / torch.sqrt(noise_level**2 + SIGMA_DATA**2)                  # [(num_samples), num_token, num_atom, 3]

        enc = self.atom_cross_att_encoder(
            batch=batch,
            token_atoms_act=act,
            trunk_single_cond=embeddings['single'],
            trunk_pair_cond=trunk_pair_cond,
        )
        act = enc.token_act                                                     # ((num_samples), num_tokens, channels)

        # Token-token attention
        single_emb = self.single_cond_embedding_projection(
            self.single_cond_embedding_norm(trunk_single_cond)
        )       
        if self.use_batch_infer:   single_emb=single_emb[None, ...]
        act += single_emb                                                       # ((num_samples), num_tokens, channels) + [(1), num_token, seq_ch]
        act = self.transformer(
            act=act,                                                            # ((num_samples), num_tokens, channels)
            single_cond=trunk_single_cond,                                      # [num_token, seq_ch]
            mask=sequence_mask,                                                 # [num_token]
            pair_cond=trunk_pair_cond,                                          # [num_token, num_token, pair_ch]
        )
        act = self.output_norm(act)                                             # ((num_samples), num_tokens, channels)

        # (Possibly) atom-granularity decoder
        position_update = self.atom_cross_att_decoder(
            batch=batch,
            token_act=act,                                                      # ((num_samples), num_tokens, channels)
            enc=enc,
        )                                                                       # ((num_samples), num_token, num_atom, 3)

        skip_scaling = SIGMA_DATA**2 / (noise_level**2 + SIGMA_DATA**2)
        out_scaling = (
            noise_level * SIGMA_DATA /
            torch.sqrt(noise_level**2 + SIGMA_DATA**2)
        )
        # End `with utils.bfloat16_context()`.
        return (
            skip_scaling * positions_noisy + out_scaling * position_update
        ) * atom_mask # ((batch), num_token, num_atom, 3) * [(1), num_token, num_atom,  1]

