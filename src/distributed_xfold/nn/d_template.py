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

from dataclasses import dataclass

import torch
import torch.nn as nn

from xfold import features, scoring, protein_data_processing
from xfold import geometry
from xfold.constants import residue_names
from xfold.nn.template import TemplateEmbedding, SingleTemplateEmbedding

from xfold import fastnn
import torch.distributed as dist
from distributed_xfold.nn import d_pairformer
from distributed_xfold.distribute_utils import shard_linear, all_gather_into_tensor, ShardInfo

from distributed_xfold.nn.template_utils import make_backbone_vectors

@dataclass
class DistogramFeaturesConfig:
    # The left edge of the first bin.
    min_bin: float = 3.25
    # The left edge of the final bin. The final bin catches everything larger than
    # `max_bin`.
    max_bin: float = 50.75
    # The number of bins in the distogram.
    num_bins: int = 39


def dgram_from_positions(positions, config: DistogramFeaturesConfig):
    """Compute distogram from amino acid positions.

    Args:
      positions: (num_res, 3) Position coordinates.
      config: Distogram bin configuration.

    Returns:
      Distogram with the specified number of bins.
    """
    lower_breaks = torch.linspace(
        config.min_bin, config.max_bin, config.num_bins, device=positions.device)
    lower_breaks = torch.square(lower_breaks)
    upper_breaks_last = torch.ones(1, device=lower_breaks.device) * 1e8
    upper_breaks = torch.concatenate(
        [lower_breaks[1:], upper_breaks_last], dim=-1
    )
    dist2 = torch.sum(
        torch.square(
            torch.unsqueeze(positions, dim=-2)
            - torch.unsqueeze(positions, dim=-3)
        ),
        dim=-1,
        keepdims=True,
    )

    dgram = (dist2 > lower_breaks).to(dtype=torch.float32) * (
        dist2 < upper_breaks
    ).to(dtype=torch.float32)
    return dgram


def make_backbone_rigid(
    positions: geometry.Vec3Array,
    mask: torch.Tensor,
    group_indices: torch.Tensor,
) -> tuple[geometry.Rigid3Array, torch.Tensor]:
    """Make backbone Rigid3Array and mask.

    Args:
      positions: (num_res, num_atoms) of atom positions as Vec3Array.
      mask: (num_res, num_atoms) for atom mask.
      group_indices: (num_res, num_group, 3) for atom indices forming groups.

    Returns:
      tuple of backbone Rigid3Array and mask (num_res,).
    """
    backbone_indices = group_indices[:, 0]

    # main backbone frames differ in sidechain frame convention.
    # for sidechain it's (C, CA, N), for backbone it's (N, CA, C)
    # Hence using c, b, a, each of shape (num_res,).
    c, b, a = [backbone_indices[..., i] for i in range(3)]
    c, b, a = [x.to(dtype=torch.int64).unsqueeze(1) for x in [c, b, a]]

    # slice_index = jax.vmap(lambda x, i: x[i])
    # rigid_mask = (
    #     slice_index(mask, a) * slice_index(mask, b) * slice_index(mask, c)
    # ).astype(jnp.float32)

    rigid_mask = torch.gather(mask, 1, a).squeeze(1) \
        * torch.gather(mask, 1, b).squeeze(1) \
        * torch.gather(mask, 1, c).squeeze(1)

    frame_positions = []
    for indices in [a, b, c]:
        frame_positions.append(
            geometry.Vec3Array(
                x=torch.gather(positions.x, 1, indices).squeeze(1),
                y=torch.gather(positions.y, 1, indices).squeeze(1),
                z=torch.gather(positions.z, 1, indices).squeeze(1),
            )
        )

    rotation = geometry.Rot3Array.from_two_vectors(
        frame_positions[2] - frame_positions[1],
        frame_positions[0] - frame_positions[1],
    )
    rigid = geometry.Rigid3Array(rotation, frame_positions[1])

    return rigid, rigid_mask.to(dtype=torch.float32)


class DistributeTemplateEmbedding(TemplateEmbedding):
    """Embed a set of templates."""

    def __init__(self, device_mesh, pair_channel: int = 128, num_channels: int = 64):
        super(DistributeTemplateEmbedding, self).__init__(pair_channel, num_channels)

        self.pair_channel = pair_channel
        self.num_channels = num_channels

        self.single_template_embedding = DistributeSingleTemplateEmbedding(device_mesh)

        self.output_linear = nn.Linear(
            self.num_channels, self.pair_channel, bias=False)

        self.rank = dist.get_rank()
        self.device_mesh = device_mesh
        self.use_dp = device_mesh.dp_size > 1
        self.dp_size = self.device_mesh.dp_size
        self.dp_shard_num = self.device_mesh.get_dp_shard_num(self.rank)
        self.dp_shard_group = self.device_mesh.get_dp_group(self.rank)

    def forward(
        self,
        query_embedding: torch.Tensor,
        templates: features.Templates,
        padding_mask_2d: torch.Tensor,
        multichain_mask_2d: torch.Tensor
    ) -> torch.Tensor:
        num_templates = templates.aatype.shape[0]
        num_res1, num_res2, _ = query_embedding.shape

        summed_template_embeddings = query_embedding.new_zeros(
            num_res1, num_res2, self.num_channels)

        if self.use_dp and num_templates >= self.dp_size:
            if self.rank == 0: print(f"Num Template: {num_templates} >= Dp size:{self.dp_size}, sharding...")
            for module in self.single_template_embedding.modules():
                if hasattr(module, 'use_dp'):
                    module.use_dp = False


            temp_shard_info = ShardInfo(num_templates, self.dp_size)
            temp_shard_slice = temp_shard_info.get_shard_slice_list()[self.dp_shard_num]

            for template_idx in range(temp_shard_slice.start, temp_shard_slice.stop):
                template_embedding = self.single_template_embedding(
                    query_embedding, templates[template_idx], padding_mask_2d, multichain_mask_2d
                )

                summed_template_embeddings += template_embedding

            dist.all_reduce(summed_template_embeddings, op=dist.ReduceOp.SUM, group=self.dp_shard_group, async_op=False)
        else:
            if self.use_dp:
                for module in self.single_template_embedding.modules():
                    if hasattr(module, 'use_dp'):
                        module.use_dp = True

            for template_idx in range(num_templates):
                template_embedding = self.single_template_embedding(
                    query_embedding, templates[template_idx], padding_mask_2d, multichain_mask_2d
                )

                summed_template_embeddings += template_embedding

        embedding = summed_template_embeddings / (1e-7 + num_templates)

        embedding = torch.relu(embedding)

        embedding = self.output_linear(embedding)

        return embedding


class DistributeSingleTemplateEmbedding(SingleTemplateEmbedding):
    """Embed a single template."""

    def __init__(self, device_mesh, num_channels: int = 64):
        super(DistributeSingleTemplateEmbedding, self).__init__(num_channels)

        self.template_embedding_iteration = nn.ModuleList(
            [d_pairformer.DistributePairformerBlock(device_mesh, c_pair=self.num_channels, num_intermediate_factor=2, with_single=False)
             for _ in range(self.template_stack_num_layer)]
        )

        self.rank = dist.get_rank()
        self.device_mesh = device_mesh
        self.use_tp = device_mesh.tp_size > 1
        self.tp_shard_group = device_mesh.get_tp_group(self.rank)

    def shard_params(self): 
        # ============ construct_input ===============
        self.template_pair_embedding_0 = shard_linear(self.template_pair_embedding_0, self.rank, self.device_mesh, True)
        self.template_pair_embedding_1 = shard_linear(self.template_pair_embedding_1, self.rank, self.device_mesh, True)
        self.template_pair_embedding_2 = shard_linear(self.template_pair_embedding_2, self.rank, self.device_mesh, True)
        self.template_pair_embedding_3 = shard_linear(self.template_pair_embedding_3, self.rank, self.device_mesh, True)
        self.template_pair_embedding_4 = shard_linear(self.template_pair_embedding_4, self.rank, self.device_mesh, True)
        self.template_pair_embedding_5 = shard_linear(self.template_pair_embedding_5, self.rank, self.device_mesh, True)
        self.template_pair_embedding_6 = shard_linear(self.template_pair_embedding_6, self.rank, self.device_mesh, True)
        self.template_pair_embedding_7 = shard_linear(self.template_pair_embedding_7, self.rank, self.device_mesh, True)
        self.template_pair_embedding_8 = shard_linear(self.template_pair_embedding_8, self.rank, self.device_mesh, True)

    def construct_input(
        self, query_embedding, templates: features.Templates, multichain_mask_2d
    ) -> torch.Tensor:
        # Compute distogram feature for the template.

        dtype = query_embedding.dtype

        aatype = templates.aatype
        dense_atom_mask = templates.atom_mask

        dense_atom_positions = templates.atom_positions
        dense_atom_positions *= dense_atom_mask[..., None]

        pseudo_beta_positions, pseudo_beta_mask = scoring.pseudo_beta_fn(
            templates.aatype, dense_atom_positions, dense_atom_mask
        )
        pseudo_beta_mask_2d = (
            pseudo_beta_mask[:, None] * pseudo_beta_mask[None, :]
        )
        pseudo_beta_mask_2d *= multichain_mask_2d
        dgram = dgram_from_positions(
            pseudo_beta_positions, self.dgram_features_config
        )
        dgram *= pseudo_beta_mask_2d[..., None]
        dgram = dgram.to(dtype=dtype)
        pseudo_beta_mask_2d = pseudo_beta_mask_2d.to(dtype=dtype)
        to_concat = [(dgram, 1), (pseudo_beta_mask_2d, 0)]

        aatype = torch.nn.functional.one_hot(
            aatype.to(dtype=torch.int64),
            residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP
        ).to(dtype=dtype)
        to_concat.append((aatype[None, :, :], 1))
        to_concat.append((aatype[:, None, :], 1))

        # Compute a feature representing the normalized vector between each
        # backbone affine - i.e. in each residues local frame, what direction are
        # each of the other residues.

        template_group_indices = torch.take_along_dim(
            protein_data_processing.RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX.to(device=templates.aatype.device),
            templates.aatype.to(dtype=torch.int64)[..., None, None],
            dim=0
        )

        unit_vector, backbone_mask = make_backbone_vectors(
            dense_atom_positions,
            dense_atom_mask,
            template_group_indices.to(dtype=torch.int32),
        )

        unit_vector = [x.to(dtype=dtype) for x in unit_vector]
        backbone_mask = backbone_mask.to(dtype=dtype)

        backbone_mask_2d = backbone_mask[:, None] * backbone_mask[None, :]
        backbone_mask_2d *= multichain_mask_2d
        unit_vector = [x * backbone_mask_2d for x in unit_vector]

        # Note that the backbone_mask takes into account C, CA and N (unlike
        # pseudo beta mask which just needs CB) so we add both masks as features.
        to_concat.extend([(x, 0) for x in unit_vector])
        to_concat.append((backbone_mask_2d, 0))

        query_embedding = self.query_embedding_norm(query_embedding)

        to_concat.append((query_embedding, 1))

        act = 0

        for i, (x, n_input_dims) in enumerate(to_concat):
            if n_input_dims == 0:
                x = x[..., None]
            act += self.__getattr__(f'template_pair_embedding_{i}')(x)

        if self.use_tp:
            unsharded_act = torch.zeros(act.shape[:-1] + (self.num_channels,), dtype=act.dtype, device=act.device)
            all_gather_into_tensor(unsharded_act, act, 
                                group=self.tp_shard_group, 
                                async_op=False,)
            act = unsharded_act                 # [N_token, N_token, pair_channel]

        return act

    def forward(
        self,
        query_embedding: torch.Tensor,
        templates: features.Templates,
        padding_mask_2d: torch.Tensor,
        multichain_mask_2d: torch.Tensor
    ) -> torch.Tensor:

        act = self.construct_input(
            query_embedding, templates, multichain_mask_2d)

        for pairformer_block in self.template_embedding_iteration:
            act = pairformer_block(act, pair_mask=padding_mask_2d)

        act = self.output_layer_norm(act)

        return act
