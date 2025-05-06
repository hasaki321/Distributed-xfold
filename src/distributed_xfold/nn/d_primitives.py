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

from xfold import fastnn
from xfold.nn.primitives import Transition, OuterProductMean
from distributed_xfold.distribute_utils import shard_linear, ShardInfo, all_gather_into_tensor, init_dist_info
import torch.distributed as dist

class DistributeTransition(Transition):

    def __init__(self, device_mesh, c_x: int, num_intermediate_factor: int = 4) -> None:
        super(DistributeTransition, self).__init__(c_x, num_intermediate_factor)
        init_dist_info(self, device_mesh)

    def shard_params(self): 
        self.transition1 = shard_linear(self.transition1, self.rank, self.device_mesh, True)
        self.transition2 = shard_linear(self.transition2, self.rank, self.device_mesh, False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer_norm(x)
        c = fastnn.gated_linear_unit(x, self.transition1.weight.T)
        c = self.transition2(c)

        if self.use_tp:
            dist.all_reduce(c, op=dist.ReduceOp.SUM, group=self.tp_shard_group)
        return c


# [TODO]
class DistributeOuterProductMean(OuterProductMean):
    def __init__(self, device_mesh, c_msa: int = 64, num_output_channel: int = 128, num_outer_channel: int = 32) -> None:
        super(DistributeOuterProductMean, self).__init__(c_msa, num_output_channel, num_outer_channel)
        
        init_dist_info(self, device_mesh)

    def shard_params(self): 
        self.right_projection = shard_linear(self.right_projection, self.rank, self.device_mesh, True)
        self.output_w = nn.Parameter(torch.chunk(self.output_w, self.tp_size, dim=1)[self.tp_shard_num])

    def forward(self, msa: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        msa : [a, b, k] | [a, d, k]
        mask : [a, b] | [a, d]
        """
        a, d, e = msa.shape
        mask = mask.unsqueeze(-1)
        msa = self.layer_norm_input(msa)

        left_act = mask * self.left_projection(msa)   
        right_act = mask * self.right_projection(msa)                               # [a, d, e // tp_size]

        left_act = left_act.permute(0, 2, 1)                                        # [a, c, b // tp_size]
        act = torch.einsum('acb,ade->dceb', left_act, right_act)                    # [d, c, e // tp_size, b // dp_size]
        act = torch.einsum('dceb,cef->dbf', act, self.output_w)                     # [d, b // dp_size, f]
        if self.use_tp:
            dist.all_reduce(act.contiguous(), op=dist.ReduceOp.SUM, group=self.tp_shard_group, async_op=False)   # [b // dp_size, d, f]
        act = act.permute(1, 0, 2) + self.output_b                                  # [b // dp_size, d, f]

        norm = torch.einsum('abc,adc->bdc', mask, mask)                       # [b // dp_size, d, f]
        return act / (self.epsilon + norm)                                          # [b // dp_size, d, f]
        
