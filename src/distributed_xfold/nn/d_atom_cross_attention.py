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


import dataclasses
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from xfold import feat_batch
from xfold.nn import atom_layout, utils
from xfold.nn.diffusion_transformer import DiffusionCrossAttTransformer

from xfold import fastnn

from xfold.nn.atom_cross_attention import (
    AtomCrossAttEncoderOutput,
    AtomCrossAttEncoder,
    AtomCrossAttDecoder
)

from distributed_xfold.nn.d_diffusion_transformer import (
    DistributeDiffusionCrossAttTransformer,
)
from distributed_xfold.distribute_utils import shard_linear, all_gather_into_tensor, init_dist_info


class DistributeAtomCrossAttEncoder(AtomCrossAttEncoder):
    def __init__(self,
                 device_mesh,
                 per_token_channels: int = 384,
                 per_atom_channels: int = 128,
                 per_atom_pair_channels: int = 16,
                 with_token_atoms_act: bool = False,
                 with_trunk_single_cond: bool = False,
                 with_trunk_pair_cond: bool = False,
                 use_batch_infer: bool = False) -> None:
        super().__init__(
                                per_token_channels, 
                                per_atom_channels, 
                                per_atom_pair_channels, 
                                with_token_atoms_act,
                                with_trunk_single_cond,
                                with_trunk_pair_cond
                            )
        self.use_batch_infer = use_batch_infer

        self.atom_transformer_encoder = DistributeDiffusionCrossAttTransformer(
            device_mesh, c_query=self.c_query, use_batch_infer=use_batch_infer)

        init_dist_info(self, device_mesh)

    def shard_params(self): 
        # ============= _per_atom_conditioning =============
        self.embed_ref_pos = shard_linear(self.embed_ref_pos, self.rank, self.device_mesh, True)
        self.embed_ref_mask = shard_linear(self.embed_ref_mask, self.rank, self.device_mesh, True)
        self.embed_ref_element = shard_linear(self.embed_ref_element, self.rank, self.device_mesh, True)
        self.embed_ref_charge = shard_linear(self.embed_ref_charge, self.rank, self.device_mesh, True)
        self.embed_ref_atom_name = shard_linear(self.embed_ref_atom_name, self.rank, self.device_mesh, True)

        self.single_to_pair_cond_row = shard_linear(self.single_to_pair_cond_row, self.rank, self.device_mesh, True)
        self.single_to_pair_cond_col = shard_linear(self.single_to_pair_cond_col, self.rank, self.device_mesh, True)
        self.embed_pair_offsets = shard_linear(self.embed_pair_offsets, self.rank, self.device_mesh, True)
        self.embed_pair_distances = shard_linear(self.embed_pair_distances, self.rank, self.device_mesh, True)

        # ============= Pair embd =============
        self.single_to_pair_cond_row_1 = shard_linear(self.single_to_pair_cond_row_1, self.rank, self.device_mesh, True)
        self.single_to_pair_cond_col_1 = shard_linear(self.single_to_pair_cond_col_1, self.rank, self.device_mesh, True)
        self.embed_pair_offsets_1 = shard_linear(self.embed_pair_offsets_1, self.rank, self.device_mesh, True)
        self.embed_pair_distances_1 = shard_linear(self.embed_pair_distances_1, self.rank, self.device_mesh, True)
        self.embed_pair_offsets_valid = shard_linear(self.embed_pair_offsets_valid, self.rank, self.device_mesh, True)

        if self.with_trunk_pair_cond is True:
            self.embed_trunk_pair_cond = shard_linear(self.embed_trunk_pair_cond, self.rank, self.device_mesh, True)

    def _per_atom_conditioning(self, batch: feat_batch.Batch) -> tuple[torch.Tensor, torch.Tensor]:

        # Compute per-atom single conditioning
        # Shape (num_tokens, num_dense, channels)
        act = self.embed_ref_pos(batch.ref_structure.positions)
        act += self.embed_ref_mask(batch.ref_structure.mask[:, :, None].to(
            dtype=self.embed_ref_mask.weight.dtype))

        # Element is encoded as atomic number if the periodic table, so
        # 128 should be fine.
        act += self.embed_ref_element(F.one_hot(batch.ref_structure.element.to(
            dtype=torch.int64), 128).to(dtype=self.embed_ref_element.weight.dtype))
        act += self.embed_ref_charge(torch.arcsinh(
            batch.ref_structure.charge)[:, :, None])

        # Characters are encoded as ASCII code minus 32, so we need 64 classes,
        # to encode all standard ASCII characters between 32 and 96.
        atom_name_chars_1hot = F.one_hot(batch.ref_structure.atom_name_chars.to(
            dtype=torch.int64), 64).to(dtype=self.embed_ref_atom_name.weight.dtype)
        num_token, num_dense, _ = act.shape
        act += self.embed_ref_atom_name(
            atom_name_chars_1hot.reshape(num_token, num_dense, -1))

        act *= batch.ref_structure.mask[:, :, None]

        if self.use_tp:
            full_act = torch.empty(act.shape[:-1] + (self.per_atom_channels, ), dtype=act.dtype, device=act.device)
            all_gather_into_tensor(
                full_act,
                act,
                group=self.tp_shard_group,
                async_op=False,
            )
            act = full_act  

        # Compute pair conditioning
        # shape (num_tokens, num_dense, num_dense, channels)
        # Embed single features
        row_act = self.single_to_pair_cond_row(torch.relu(act))
        col_act = self.single_to_pair_cond_col(torch.relu(act))
        pair_act = row_act[:, :, None, :] + col_act[:, None, :, :]

        # Embed pairwise offsets
        pair_act += self.embed_pair_offsets(batch.ref_structure.positions[:, :, None, :]
                                            - batch.ref_structure.positions[:, None, :, :])

        sq_dists = torch.sum(
            torch.square(
                batch.ref_structure.positions[:, :, None, :]
                - batch.ref_structure.positions[:, None, :, :]
            ),
            dim=-1,
        )
        pair_act += self.embed_pair_distances(1.0 /
                                              (1 + sq_dists[:, :, :, None]))

        if self.use_tp:
            full_pair_act = torch.empty(pair_act.shape[:-1] + (self.per_atom_pair_channels, ), dtype=pair_act.dtype, device=pair_act.device)
            all_gather_into_tensor(
                full_pair_act,
                pair_act,
                group=self.tp_shard_group,
                async_op=False,
            )
            pair_act = full_pair_act  

        return act, pair_act

    # @torch.compiler.disable
    def forward(
        self,
        batch: feat_batch.Batch,
        token_atoms_act: torch.Tensor | None,  # ((batch), num_tokens, max_atoms_per_token, 3)
        trunk_single_cond: torch.Tensor | None,  # (num_tokens, ch)
        trunk_pair_cond: torch.Tensor | None,  # (num_tokens, num_tokens, ch)
    ) -> AtomCrossAttEncoderOutput:
        """Cross-attention on flat atom subsets and mapping to per-token features."""

        # Compute single conditioning from atom meta data and convert to queries layout.
        token_atoms_single_cond, _ = self._per_atom_conditioning(batch)         # (num_tokens, max_atoms_per_token, channels // tp_size)
        token_atoms_mask = batch.predicted_structure_info.atom_mask             # (num_tokens, max_atoms_per_token）
        queries_single_cond = atom_layout.convert(                              # (num_res, num_query, channels // tp_siz)
            batch.atom_cross_att.token_atoms_to_queries,
            token_atoms_single_cond,
            layout_axes=(-3, -2),
        )                                                                      

        queries_mask = atom_layout.convert(                                     # (num_res, num_query)
            batch.atom_cross_att.token_atoms_to_queries,
            token_atoms_mask,
            layout_axes=(-2, -1),
        )

        if trunk_single_cond is not None:
            trunk_single_cond = self.embed_trunk_single_cond(                   # (num_tokens, channels)
                self.lnorm_trunk_single_cond(trunk_single_cond)
                ) 
            queries_single_cond += atom_layout.convert(                         # (num_res, num_query, channels) + (num_res, num_query, channels)
                batch.atom_cross_att.tokens_to_queries,
                trunk_single_cond,
                layout_axes=(-2,),
            )

        if token_atoms_act is None:
            queries_act = queries_single_cond.clone()                           # (num_res, num_query, channels)
        else:
            # Convert token_atoms_act to queries layout and map to per_atom_channels
            # (num_subsets, num_queries, channels)  
            queries_act = atom_layout.convert(                                  # ((batch), num_tokens, max_atoms_per_token, 3)
                batch.atom_cross_att.token_atoms_to_queries,
                token_atoms_act,  
                layout_axes=(-3, -2),
            )                                                                   # ((batch), num_res, num_query, 3)
            queries_act = self.atom_positions_to_features(queries_act)          # ((batch), num_res, num_query, channels)

            if self.use_batch_infer:
                queries_act *= queries_mask[None, ..., None]    # (batch, num_res, num_query, channels) *  (1, num_res, num_query, 1)
                queries_act += queries_single_cond[None, ...]   # (batch, num_res, num_query, channels) +  (1, num_res, num_query, channels)
            else:
                queries_act *= queries_mask[..., None]          # (num_res, num_query, channels) *  (num_res, num_query, 1)
                queries_act += queries_single_cond              # (num_res, num_query, channels) +  (num_res, num_query, channels)
            
            # Gather the keys from the queries.
        keys_single_cond = atom_layout.convert(
            batch.atom_cross_att.queries_to_keys,
            queries_single_cond,
            layout_axes=(-3, -2),
        )                                                                       # (num_res, num_key, channels) 
        keys_mask = atom_layout.convert(
            batch.atom_cross_att.queries_to_keys, queries_mask, layout_axes=(-2, -1)
        )                                                                       # (num_res, num_key, ) 

        # Embed single features into the pair conditioning.



        row_act = self.single_to_pair_cond_row_1(F.relu(queries_single_cond))   # ((batch), num_res, num_query, channels // tp_size)

        pair_cond_keys_input = atom_layout.convert(
                    batch.atom_cross_att.queries_to_keys,
                    queries_single_cond,
                    layout_axes=(-3, -2), 
                )                                                               # ((batch), num_res, num_key, channels // tp_size)
        
        col_act = self.single_to_pair_cond_col_1(F.relu(pair_cond_keys_input))  # (num_res, num_key, channels // tp_size)
        pair_act = row_act[:, :, None, :] + col_act[:, None, :, :]              # (num_res, num_query, num_key, channels // tp_size)

        # (num_tokens, num_tokens, per_atom_pair_channels)
        if trunk_pair_cond is not None:
            trunk_pair_cond = self.embed_trunk_pair_cond(self.lnorm_trunk_pair_cond(trunk_pair_cond)) # (num_tokens, num_tokens, channels // tp_size)
            num_tokens = trunk_pair_cond.shape[0]
            tokens_to_queries = batch.atom_cross_att.tokens_to_queries
            tokens_to_keys = batch.atom_cross_att.tokens_to_keys
            trunk_pair_to_atom_pair = atom_layout.GatherInfo(
                gather_idxs=(
                    num_tokens * tokens_to_queries.gather_idxs[:, :, None]
                    + tokens_to_keys.gather_idxs[:, None, :]
                ),
                gather_mask=(
                    tokens_to_queries.gather_mask[:, :, None]
                    & tokens_to_keys.gather_mask[:, None, :]
                ),
                input_shape=torch.tensor((num_tokens, num_tokens), device=torch.device('cpu')),
            )
            pair_act += atom_layout.convert(
                trunk_pair_to_atom_pair, trunk_pair_cond, layout_axes=(-3, -2)
            ) # (num_res, num_query, num_key, channels // tp_size) +  (num_res, num_query, num_key, channels // tp_size)

        # Embed pairwise offsets
        queries_ref_pos = atom_layout.convert(
            batch.atom_cross_att.token_atoms_to_queries,
            batch.ref_structure.positions,
            layout_axes=(-3, -2),
        )
        queries_ref_space_uid = atom_layout.convert(
            batch.atom_cross_att.token_atoms_to_queries,
            batch.ref_structure.ref_space_uid,
            layout_axes=(-2, -1),
        )
        keys_ref_pos = atom_layout.convert(
            batch.atom_cross_att.queries_to_keys,
            queries_ref_pos,
            layout_axes=(-3, -2),
        )
        keys_ref_space_uid = atom_layout.convert(
            batch.atom_cross_att.queries_to_keys,
            batch.ref_structure.ref_space_uid,
            layout_axes=(-2, -1),
        )

        offsets_valid = (
            queries_ref_space_uid[:, :, None] == keys_ref_space_uid[:, None, :]
        )
        offsets = queries_ref_pos[:, :, None, :] - keys_ref_pos[:, None, :, :]
        pair_act += (
            self.embed_pair_offsets_1(offsets)                                  # (num_res, num_query, num_key, channels // tp_size)
            * offsets_valid[:, :, :, None]                                      # (num_res, num_query, num_key, 1)
        )# (num_res, num_query, num_key, channels // tp_size) +  (num_res, num_query, num_key, channels // tp_size)

        # Embed pairwise inverse squared distances
        sq_dists = torch.sum(torch.square(offsets), dim=-1) # (num_res, num_query, num_key, )
        pair_act += (
            self.embed_pair_distances_1(1.0 / (1 + sq_dists[:, :, :, None]))    # (num_res, num_query, num_key, channels // tp_size)
            * offsets_valid[:, :, :, None]                                      # (num_res, num_query, num_key, 1)
        )  # (num_res, num_query, num_key, channels // tp_size) +  (num_res, num_query, num_key, channels // tp_size)

        # Embed offsets valid mask
        pair_act += self.embed_pair_offsets_valid(offsets_valid[:, :, :, None].to(
            dtype=self.embed_pair_offsets_valid.weight.dtype))
        # (num_res, num_query, num_key, channels // tp_size) +  (num_res, num_query, num_key, channels // tp_size)

        if self.use_tp:
            full_pair_act = torch.empty((pair_act.shape[:-1] + (self.per_atom_pair_channels,)), dtype=pair_act.dtype, device=queries_single_cond.device)
            all_gather_into_tensor(
                full_pair_act,
                pair_act,
                group=self.tp_shard_group,
                async_op=False,
            )
            pair_act = full_pair_act
        pair_act2 = self.pair_mlp_1(torch.relu(pair_act))
        pair_act2 = self.pair_mlp_2(torch.relu(pair_act2))
        pair_act += self.pair_mlp_3(torch.relu(pair_act2)) # (num_res, num_query, num_key, channels)

        # print(pair_act.shape)
        # Run the atom cross attention transformer.
        queries_act = self.atom_transformer_encoder(
            queries_act=queries_act,                                # ((batch), num_res, num_query, channels)
            queries_mask=queries_mask,                              # (num_res, num_query)
            queries_to_keys=batch.atom_cross_att.queries_to_keys,   # (num_res, num_key)
            keys_mask=keys_mask,                                    # (num_res, num_key,)
            queries_single_cond=queries_single_cond,                # (num_res, num_query, channels)
            keys_single_cond=keys_single_cond,                      # (num_res, num_key, channels)
            pair_cond=pair_act,                                     # (num_res, num_query, num_key, channels)
        )                                                           # ((batch), num_res, num_query, channels)
        if self.use_batch_infer:
            queries_act *= queries_mask[None, ..., None]    # (batch, num_res, num_query, channels) *  (1, num_res, num_query, 1)
        else:
            queries_act *= queries_mask[..., None]          # (num_res, num_query, channels) *  (num_res, num_query, 1)

        skip_connection = queries_act

        # Convert back to token-atom layout and aggregate to tokens
        queries_act = self.project_atom_features_for_aggr(queries_act)
        token_atoms_act = atom_layout.convert(
            batch.atom_cross_att.queries_to_token_atoms,
            queries_act,
            layout_axes=(-3, -2),
        ) # (batch, num_tokens, max_atoms_per_token， channels)
        token_act = utils.mask_mean(
            token_atoms_mask[None, ..., None] if self.use_batch_infer else token_atoms_mask[..., None], 
            F.relu(token_atoms_act), dim=-2
        ) # (1, num_tokens, max_atoms_per_token， 1）, (batch, num_tokens, max_atoms_per_token， channels)

        return AtomCrossAttEncoderOutput(
            token_act=token_act,                                        # (batch, num_tokens, channels)
            skip_connection=skip_connection,                            # (batch, num_res, num_query, channels)
            queries_mask=queries_mask,                                  # (num_res, num_query)
            queries_single_cond=queries_single_cond,                    # (num_res, num_query, channels)
            keys_mask=keys_mask,                                        # (num_res, num_key, )
            keys_single_cond=keys_single_cond,                          # (num_res, num_key, channels)
            pair_cond=pair_act,                                         # (num_res, num_query, num_key, channels)
        )


class DistributeAtomCrossAttDecoder(AtomCrossAttDecoder):
    def __init__(self, device_mesh, use_batch_infer:bool = False) -> None:
        super().__init__()
        self.use_batch_infer = use_batch_infer
        self.atom_transformer_decoder = DistributeDiffusionCrossAttTransformer(
            device_mesh, c_query=self.per_atom_channels, use_batch_infer=use_batch_infer)

    # @torch.compiler.disable
    def forward(self,
        batch: feat_batch.Batch,
        token_act: torch.Tensor,  # (batch, num_tokens, channels)
        enc: AtomCrossAttEncoderOutput) -> torch.Tensor:
        """Mapping to per-atom features and self-attention on subsets."""

        # Map per-token act down to per_atom channels
        token_act = self.project_token_features_for_broadcast(token_act)

        # Broadcast to token-atoms layout and convert to queries layout
        num_token, max_atoms_per_token = batch.atom_cross_att.queries_to_token_atoms.shape
        if self.use_batch_infer:
            token_atom_act = token_act[:, :, None, :].expand(
                token_act.shape[0], num_token, max_atoms_per_token, self.per_atom_channels
            )                                                                       # (batch, num_tokens, num_atom, channels)
        else:
            token_atom_act = token_act[:, None, :].expand(
                num_token, max_atoms_per_token, self.per_atom_channels
            )                                                                       # (num_tokens, num_atom, channels)
        queries_act = atom_layout.convert(
            batch.atom_cross_att.token_atoms_to_queries,
            token_atom_act,
            layout_axes=(-3, -2),
        )                                                                       # ((batch), num_res, num_query, channels)

        # Add skip connection from encoder
        queries_act += enc.skip_connection                  # ((batch), num_res, num_query, channels) + ((batch), num_res, num_query, channels)
        q_mask = enc.queries_mask[None, ..., None] if self.use_batch_infer else enc.queries_mask[..., None]
        queries_act *= q_mask                               # ((batch), num_res, num_query, channels) * ((1), num_res, num_query, 1 )

        # Run the atom cross attention transformer
        
        queries_act = self.atom_transformer_decoder(
            queries_act=queries_act,                                            # ((batch), num_res, num_query, channels)
            queries_mask=enc.queries_mask,                                      # (num_res, num_query)
            queries_to_keys=batch.atom_cross_att.queries_to_keys,               # (num_res, num_key)
            keys_mask=enc.keys_mask,                                            # (num_res, num_key, )
            queries_single_cond=enc.queries_single_cond,                        # (num_res, num_query, channels)
            keys_single_cond=enc.keys_single_cond,                              # (num_res, num_key, channels)
            pair_cond=enc.pair_cond,                                            # (num_res, num_query, num_key, channels)
        )
        queries_act *= q_mask                               # ((batch), num_res, num_query, channels) * ((1), num_res, num_query, 1 )

        # Apply layer normalization
        queries_act = self.atom_features_layer_norm(queries_act)

        # Map atom features to position updates
        queries_position_update = self.atom_features_to_position_update(queries_act)

        # Convert back to token-atoms layout
        position_update = atom_layout.convert(
            batch.atom_cross_att.queries_to_token_atoms,
            queries_position_update,
            layout_axes=(-3, -2),
        )                                                                       # ((batch), num_token, num_atom, 3)
        return position_update