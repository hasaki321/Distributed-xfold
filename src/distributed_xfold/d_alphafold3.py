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

from xfold import feat_batch, features
from xfold.nn import featurization
from xfold.nn.head import DistogramHead, ConfidenceHead
from xfold.nn import atom_cross_attention, diffusion_head
from xfold.nn import diffusion_head

from xfold.alphafold3 import Evoformer

# from distributed_xfold.nn import atom_cross_attention_batch, diffusion_head_batch
from distributed_xfold.nn import d_diffusion_head
from distributed_xfold.distribute_utils import all_gather_into_tensor, ShardInfo, shard_linear, DeviceMesh
from distributed_xfold.nn.d_template import DistributeTemplateEmbedding
from distributed_xfold.nn.d_pairformer import DistributeEvoformerBlock, DistributePairformerBlock
from distributed_xfold.nn import d_atom_cross_attention

import torch.distributed as dist
import os

class DistributeEvoformer(Evoformer):
    def __init__(self, device_mesh, msa_channel: int = 64):
        super(DistributeEvoformer, self).__init__(msa_channel)

        self.template_embedding = DistributeTemplateEmbedding(
            device_mesh, pair_channel=self.pair_channel)

        self.msa_stack = nn.ModuleList(
            [DistributeEvoformerBlock(device_mesh) for _ in range(self.msa_stack_num_layer)])

        self.trunk_pairformer = nn.ModuleList(
            [DistributePairformerBlock(device_mesh, with_single=True) for _ in range(self.pairformer_num_layer)])

        # =============== Distribute Informations ===================
        self.rank = dist.get_rank()
        self.device_mesh = device_mesh
        self.use_tp = device_mesh.tp_size > 1
        self.dp_size = self.device_mesh.dp_size
        self.tp_size = self.device_mesh.tp_size
        self.dp_shard_num = self.device_mesh.get_dp_shard_num(self.rank)
        self.tp_shard_num = self.device_mesh.get_tp_shard_num(self.rank)
        self.dp_shard_group = self.device_mesh.get_dp_group(self.rank)
        self.tp_shard_group = self.device_mesh.get_tp_group(self.rank)
    
    def shard_params(self): 
        # ========== _seq_pair_embedding ===========
        self.left_single = shard_linear(self.left_single, self.rank, self.device_mesh, True)
        self.right_single = shard_linear(self.right_single, self.rank, self.device_mesh, True)

        self.prev_embedding = shard_linear(self.prev_embedding, self.rank, self.device_mesh, True)
        
        # ========== _relative_encoding ===========
        self.position_activations = shard_linear(self.position_activations, self.rank, self.device_mesh, True)

        # ========== _embed_bonds ===========
        self.bond_embedding = shard_linear(self.bond_embedding, self.rank, self.device_mesh, True)

        # ========== single emb ===========
        self.single_activations = shard_linear(self.single_activations, self.rank, self.device_mesh, True)
        self.prev_single_embedding = shard_linear(self.prev_single_embedding, self.rank, self.device_mesh, True)

    # Tensor parallel for pair
    def _seq_pair_embedding(
        self, token_features: features.TokenFeatures, target_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generated Pair embedding from sequence."""
        left_single = self.left_single(target_feat)[:, None]
        right_single = self.right_single(target_feat)[None]
        pair_activations = left_single + right_single

        mask = token_features.mask

        pair_mask = (mask[:, None] * mask[None, :]).to(dtype=left_single.dtype)

        return pair_activations, pair_mask

    def _relative_encoding(
        self, batch: feat_batch.Batch, pair_activations: torch.Tensor
    ) -> torch.Tensor:
        max_relative_idx = 32
        max_relative_chain = 2

        rel_feat = featurization.create_relative_encoding(
            batch.token_features,
            max_relative_idx,
            max_relative_chain,
        ).to(dtype=pair_activations.dtype)

        pair_activations += self.position_activations(rel_feat)
        return pair_activations

    def _embed_bonds(
        self, batch: feat_batch.Batch, pair_activations: torch.Tensor
    ) -> torch.Tensor:
        """Embeds bond features and merges into pair activations."""
        # Construct contact matrix.
        
        num_tokens = batch.token_features.token_index.shape[0]
        contact_matrix = torch.zeros(
            (num_tokens, num_tokens), dtype=pair_activations.dtype, device=pair_activations.device)

        tokens_to_polymer_ligand_bonds = (
            batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds
        )
        gather_idxs_polymer_ligand = tokens_to_polymer_ligand_bonds.gather_idxs
        gather_mask_polymer_ligand = (
            tokens_to_polymer_ligand_bonds.gather_mask.prod(dim=1).to(
                dtype=gather_idxs_polymer_ligand.dtype)[:, None]
        )
        # If valid mask then it will be all 1's, so idxs should be unchanged.
        gather_idxs_polymer_ligand = (
            gather_idxs_polymer_ligand * gather_mask_polymer_ligand
        )

        tokens_to_ligand_ligand_bonds = (
            batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds
        )
        gather_idxs_ligand_ligand = tokens_to_ligand_ligand_bonds.gather_idxs
        gather_mask_ligand_ligand = tokens_to_ligand_ligand_bonds.gather_mask.prod(
            dim=1
        ).to(dtype=gather_idxs_ligand_ligand.dtype)[:, None]
        gather_idxs_ligand_ligand = (
            gather_idxs_ligand_ligand * gather_mask_ligand_ligand
        )

        gather_idxs = torch.concatenate(
            [gather_idxs_polymer_ligand, gather_idxs_ligand_ligand]
        )
        contact_matrix[
            gather_idxs[:, 0], gather_idxs[:, 1]
        ] = 1.0

        # Because all the padded index's are 0's.
        contact_matrix[0, 0] = 0.0

        bonds_act = self.bond_embedding(contact_matrix[:, :, None])

        return pair_activations + bonds_act

    def _embed_template_pair(
        self,
        batch: feat_batch.Batch,
        pair_activations: torch.Tensor,
        pair_mask: torch.Tensor
    ) -> torch.Tensor:
        """Embeds Templates and merges into pair activations."""
        templates = batch.templates
        asym_id = batch.token_features.asym_id

        dtype = pair_activations.dtype
        multichain_mask = (asym_id[:, None] ==
                           asym_id[None, :]).to(dtype=dtype)

        template_act = self.template_embedding(
            query_embedding=pair_activations,
            templates=templates,
            multichain_mask_2d=multichain_mask,
            padding_mask_2d=pair_mask
        )

        return pair_activations + template_act

    def _embed_process_msa(
        self, msa_batch: features.MSA,
        pair_activations: torch.Tensor,
        pair_mask: torch.Tensor,
        target_feat: torch.Tensor
    ) -> torch.Tensor:
        """Processes MSA and returns updated pair activations."""
        dtype = pair_activations.dtype

        msa_batch = featurization.shuffle_msa(msa_batch)
        msa_batch = featurization.truncate_msa_batch(msa_batch, self.num_msa)

        msa_mask = msa_batch.mask.to(dtype=dtype)
        msa_feat = featurization.create_msa_feat(msa_batch).to(dtype=dtype)

        msa_activations = self.msa_activations(msa_feat)
        msa_activations += self.extra_msa_target_feat(target_feat)[None]

        # Evoformer MSA stack.
        for msa_block in self.msa_stack:
            msa_activations, pair_activations = msa_block(
                msa=msa_activations,
                pair=pair_activations,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
            )

        return pair_activations

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        prev: dict[str, torch.Tensor],
        target_feat: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        pair_activations, pair_mask = self._seq_pair_embedding(
            batch.token_features, target_feat
        )# [N_token, N_token, pair_channel // tp_size], [N_token, N_token]
        pair_activations += self.prev_embedding(
            self.prev_embedding_layer_norm(prev['pair'])) # [N_token, N_token, pair_channel // tp_size]

        pair_activations = self._relative_encoding(batch, pair_activations)  # [N_token, N_token, pair_channel // tp_size]

        pair_activations = self._embed_bonds(                # [N_token, N_token, pair_channel // tp_size]
            batch=batch, pair_activations=pair_activations
        )

        if self.use_tp:
            unsharded_pair_activations = torch.zeros(pair_activations.shape[:-1] + (self.pair_channel,), dtype=pair_activations.dtype, device=pair_activations.device)
            all_gather_into_tensor(unsharded_pair_activations, pair_activations, 
                                group=self.tp_shard_group, 
                                async_op=False,)
            pair_activations = unsharded_pair_activations       # [N_token, N_token, pair_channel]
        pair_activations = self._embed_template_pair(           # [N_token, N_token, pair_channel]
            batch=batch,
            pair_activations=pair_activations,
            pair_mask=pair_mask,
        )
        pair_activations = self._embed_process_msa(             # [N_token, N_token, pair_channel]
            msa_batch=batch.msa,
            pair_activations=pair_activations,
            pair_mask=pair_mask,
            target_feat=target_feat,
        )
        single_activations = self.single_activations(target_feat)
        single_activations += self.prev_single_embedding(
            self.prev_single_embedding_layer_norm(prev['single']))
        if self.use_tp:
            unsharded_single_activations = torch.zeros(single_activations.shape[:-1] + (self.seq_channel,), dtype=single_activations.dtype, device=single_activations.device)
            all_gather_into_tensor(unsharded_single_activations, single_activations, 
                                group=self.tp_shard_group, 
                                async_op=False,)
            single_activations = unsharded_single_activations       # [N_token, seq_channel]

        # TODO: Make sure the full copy of input here
        for pairformer_b in self.trunk_pairformer:
            pair_activations, single_activations = pairformer_b(
                pair_activations, pair_mask, single_activations, batch.token_features.mask)

        output = {
            'single': single_activations,
            'pair': pair_activations,
            'target_feat': target_feat,
        }

        return output

class DistributeAlphaFold3(nn.Module):
    def __init__(self, device_mesh, num_recycles: int = 10, num_samples: int = 5, diffusion_steps: int = 200):
        super(DistributeAlphaFold3, self).__init__()

        self.num_recycles = num_recycles
        self.num_samples = num_samples
        self.diffusion_steps = diffusion_steps

        self.gamma_0 = 0.8
        self.gamma_min = 1.0
        self.noise_scale = 1.003
        self.step_scale = 1.5

        self.evoformer_pair_channel = 128
        self.evoformer_seq_channel = 384

        evo_device_mesh = DeviceMesh(device_mesh.world_size, dp_size=device_mesh.world_size, tp_size=1)
        self.evoformer = DistributeEvoformer(evo_device_mesh)

        self.batch_infer = num_samples!=1 and (num_samples != device_mesh.dp_size)

        self.evoformer_conditioning = d_atom_cross_attention.DistributeAtomCrossAttEncoder(device_mesh)

        self.diffusion_head = d_diffusion_head.DistributeDiffusionHead(device_mesh, use_batch_infer=self.batch_infer)

        self.distogram_head = DistogramHead()
        self.confidence_head = ConfidenceHead()
        
        n_pairformer_layers = 4
        self.confidence_head.confidence_pairformer = nn.ModuleList([
            DistributePairformerBlock(
                device_mesh,
                c_single=self.confidence_head.c_single,
                c_pair=self.confidence_head.c_pair,
                with_single=True,
            ) for _ in range(n_pairformer_layers)
        ])
        for module in self.confidence_head.confidence_pairformer.modules():
            if hasattr(module, 'use_dp'):
                module.use_dp = False

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

    # @torch.compiler.disable()
    def create_target_feat_embedding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        target_feat = featurization.create_target_feat(
            batch,
            append_per_atom_features=False,
        )

        enc = self.evoformer_conditioning(
            token_atoms_act=None,
            trunk_single_cond=None,
            trunk_pair_cond=None,
            batch=batch,
        )

        target_feat = torch.concatenate([target_feat, enc.token_act], dim=-1)

        return target_feat

    def _sample_diffusion(
        self,
        batch: feat_batch.Batch,
        embeddings: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Sample using denoiser on batch."""

        mask = batch.predicted_structure_info.atom_mask

        def _apply_denoising_step(
            positions: torch.Tensor,
            noise_level: torch.Tensor,
            t_hat: torch.Tensor,
            noise_scale: torch.Tensor,
            trunk_single_cond: torch.Tensor,
            trunk_pair_cond: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:

            positions = diffusion_head.random_augmentation(
                positions=positions, mask=(mask[None, ...] if self.batch_infer else mask)
            )
            # noise = noise_scale * torch.randn(size=positions.shape, device=noise_scale.device)
            # noise = noise_scale
            positions_noisy = positions + noise_scale

            positions_denoised = self.diffusion_head(positions_noisy=positions_noisy,
                                                    noise_level=t_hat,
                                                    batch=batch,
                                                    embeddings=embeddings,
                                                    trunk_single_cond=trunk_single_cond,
                                                    trunk_pair_cond=trunk_pair_cond)
            grad = (positions_noisy - positions_denoised) / t_hat

            d_t = noise_level - t_hat
            positions_out = positions_noisy + self.step_scale * d_t * grad

            return positions_out


        num_samples = self.num_samples

        device = mask.device

        noise_levels = diffusion_head.noise_schedule(
            torch.linspace(0, 1, self.diffusion_steps + 1, device=device))
        
        noise_level_prev = noise_levels[:-1]
        noise_level = noise_levels[1:]

        gamma = self.gamma_0 * (noise_level > self.gamma_min)
        t_hat = noise_level_prev * (1 + gamma)

        noise_scale = self.noise_scale * torch.sqrt(t_hat**2 - noise_level_prev**2)

        # Get conditioning first to aviod redundent recomputation
        trunk_single_cond, trunk_pair_cond = self.diffusion_head.pre_conditioning(
            batch=batch,
            embeddings=embeddings,
            noise_level=t_hat,
            use_conditioning=True,
        ) # [t, num_token, seq_ch], [num_token, num_token, pair_ch]

        if not self.batch_infer:
            positions = torch.randn(mask.shape + (3,), device=device)
        else:
            if self.use_dp:
                num_sample_shard_info = ShardInfo(num_samples, self.dp_size)
                num_sample_shard_sizes = num_sample_shard_info.get_shard_size_list()
                positions = torch.randn(
                    (num_sample_shard_sizes[self.dp_shard_num],) + mask.shape + (3,), device=device)
            else:
                positions = torch.randn(
                    (num_samples,) + mask.shape + (3,), device=device)
        positions *= noise_levels[0]

        for step_idx in range(self.diffusion_steps):

            positions = _apply_denoising_step(
                    positions, 
                    noise_level[step_idx],
                    t_hat[step_idx],
                    noise_scale[step_idx],
                    trunk_single_cond[step_idx],
                    trunk_pair_cond
                )
            if self.rank == 0: print(f"{step_idx}-th loop for diffusion")
        
        if not self.batch_infer: positions = positions[None, ...]
        return {'atom_positions': positions, 'mask': mask}

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = feat_batch.Batch.from_data_dict(batch)
        num_res = batch.num_res

        target_feat = self.create_target_feat_embedding(batch)

        embeddings = {
            'pair': torch.zeros(
                [num_res, num_res, self.evoformer_pair_channel], device=target_feat.device,
                dtype=torch.float32,
            ),
            'single': torch.zeros(
                [num_res, self.evoformer_seq_channel], dtype=torch.float32, device=target_feat.device,
            ),
            'target_feat': target_feat,  # type: ignore
        }

        origin_interop_threads = torch.get_num_interop_threads()
        origin_intraop_threads = torch.get_num_threads()
        # torch.set_num_interop_threads(1)

        evo_intra_threads = int(os.getenv('EVO_INTRA_THREAD', origin_intraop_threads))
        torch.set_num_threads(evo_intra_threads) # 40

        if self.rank == 0:
            print(f"Start Evoformer using {torch.get_num_interop_threads()}-inter-threads")
            print(f"Start Evoformer using {torch.get_num_threads()}-intra-threads")
        for i in range(self.num_recycles):
            embeddings = self.evoformer(
                batch=batch,
                prev=embeddings,
                target_feat=target_feat
            )
            if self.rank == 0:
                print(f"{i}-th loop for evoformer")

        torch.set_num_interop_threads(origin_interop_threads)
        torch.set_num_threads(origin_intraop_threads)
        if self.rank == 0:
            print(f"Start Diffusion using {torch.get_num_interop_threads()}-inter-threads")
            print(f"Start Diffusion using {torch.get_num_threads()}-intra-threads")

        samples = self._sample_diffusion(batch, embeddings)

        confidence_output_per_sample = []
        for sample_dense_atom_position in samples['atom_positions']:
            confidence_output_per_sample.append(self.confidence_head(
                dense_atom_positions=sample_dense_atom_position,
                embeddings=embeddings,
                seq_mask=batch.token_features.mask,
                token_atoms_to_pseudo_beta=batch.pseudo_beta_info.token_atoms_to_pseudo_beta,
                asym_id=batch.token_features.asym_id
            ))
        
        if self.use_dp:
            full_confidence_output_per_sample = [None for _ in range(ShardInfo(self.num_samples, self.dp_size).get_num_shard())]
            dist.all_gather_object(full_confidence_output_per_sample, confidence_output_per_sample, group=self.dp_shard_group)
            confidence_output_per_sample = sum(full_confidence_output_per_sample, [])

            positions = samples['atom_positions']
            full_positions = torch.randn((self.num_samples,) + (positions.shape[1:]), device=positions.device, dtype=positions.dtype)
            all_gather_into_tensor(
                full_positions,
                positions,
                group=self.dp_shard_group, async_op=False
            )
            samples['atom_positions'] = full_positions

        confidence_output = {}
        for key in confidence_output_per_sample[0]:
            confidence_output[key] = torch.stack([sample[key] for sample in confidence_output_per_sample], dim=0)
        samples['mask'] = torch.tile(samples['mask'][None], (self.num_samples, 1, 1))

        distogram = self.distogram_head(batch, embeddings)

        return {
            'diffusion_samples': samples,
            'distogram': distogram,
            **confidence_output,
        }
