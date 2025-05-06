###############################################################################
# Copyright (c) 2023 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Author: Narendra Chaudhary (Intel Corp.)                                    #
###############################################################################


import math
import torch
from torch import nn
from torch.autograd import Function

from . import TRI_BLOCKSIZE, QKV_BLOCKSIZE, A_BLOCKSIZE, Ak_BLOCKSIZE, C_BLOCKSIZE

class TriangleMultiplicationFunction(Function):
    @staticmethod
    def forward(
        ctx,
        act,
        mask,
        c_equation,
        left_norm_input_weight,
        left_norm_input_bias,
        projection_weight,
        gate_weight,
        center_norm_weight,
        center_norm_bias,
        output_projection_weight,
        gating_linear_weight,
    ):
        equation_flag = int(0)
        if c_equation == "ikc,jkc->ijc":  # "Outgoing" edges equation
            equation_flag = 0
        else:  # "Incoming" edges equation
            equation_flag = 1
        Bp_t, Sp_t, H = act.shape
        act = torch.ops._alpha_attention.traingle_multiplication_forward(
            act.contiguous(),
            mask.contiguous(),
            equation_flag,
            left_norm_input_weight,
            left_norm_input_bias,
            projection_weight,
            gate_weight,
            center_norm_weight,
            center_norm_bias,
            output_projection_weight,
            gating_linear_weight,
        )
        B_t, S_t, H = act.shape
        if B_t != Bp_t:
            act = act.narrow(0, 0, Bp_t)
        if S_t != Sp_t:
            act = act.narrow(1, 0, Sp_t)
        return act


def TriangleMultiplicationXSMM_forward(self, act, mask):
    mask = mask[..., None].float()

    if (act.dtype == torch.bfloat16):
        assert  mask.dtype == torch.float32 \
            and self.left_norm_input.weight.dtype == torch.bfloat16 \
            and self.left_norm_input.bias.dtype == torch.bfloat16 \
            and self.projection.weight.dtype == torch.bfloat16 \
            and self.gate.weight.dtype == torch.bfloat16 \
            and self.center_norm.weight.dtype == torch.bfloat16 \
            and self.center_norm.bias.dtype == torch.bfloat16 \
            and self.output_projection.weight.dtype == torch.bfloat16 \
            and self.gating_linear.weight.dtype == torch.bfloat16
            
    act = TriangleMultiplicationFunction.apply(
        act,
        mask,
        self.c_equation,
        self.left_norm_input.weight,
        self.left_norm_input.bias,
        self.projection.weight,
        self.gate.weight,
        self.center_norm.weight,
        self.center_norm.bias,
        self.output_projection.weight,
        self.gating_linear.weight,
    )
    return act


class TriangleMultiplicationXSMM(nn.Module):

    #   def __init__(self,config, global_config, act_dim):
    def __init__(self, equation, num_intermediate_channel, act_dim):
        """Builds TriangleMultiplication module.

        Arguments:
          act: Pair activations, shape [N_res, N_res, c_z]
          mask: Pair mask, shape [N_res, N_res].
          is_training: Whether the module is in training mode.

        Returns:
          Outputs, same shape/type as act.
        """
        super().__init__()
        # self.config = config
        # self.global_config = global_config
        # self.c_equation = self.config['equation']
        self.c_equation = equation
        # self.num_intermediate_channel = num_intermediate_channel
        self.left_norm_input = nn.LayerNorm(act_dim)
        self.projection = nn.Linear(act_dim, 2 * num_intermediate_channel)
        self.gate = nn.Linear(num_intermediate_channel, 2 * num_intermediate_channel)
        self.center_norm = nn.LayerNorm(num_intermediate_channel)
        self.output_projection = nn.Linear(num_intermediate_channel, act_dim)
        self.gating_linear = nn.Linear(act_dim, act_dim)

    def forward(self, act, mask):
        mask = mask[..., None].float()

        if (act.dtype == torch.bfloat16):
            assert  mask.dtype == torch.float32 \
                and self.left_norm_input.weight.dtype == torch.bfloat16 \
                and self.left_norm_input.bias.dtype == torch.bfloat16 \
                and self.projection.weight.dtype == torch.bfloat16 \
                and self.gate.weight.dtype == torch.bfloat16 \
                and self.center_norm.weight.dtype == torch.bfloat16 \
                and self.center_norm.bias.dtype == torch.bfloat16 \
                and self.output_projection.weight.dtype == torch.bfloat16 \
                and self.gating_linear.weight.dtype == torch.bfloat16
            
        act = TriangleMultiplicationFunction.apply(
            act,
            mask.float(),
            self.c_equation,
            self.left_norm_input.weight,
            self.left_norm_input.bias,
            self.projection.weight,
            self.gate.weight,
            self.center_norm.weight,
            self.center_norm.bias,
            self.output_projection.weight,
            self.gating_linear.weight,
        )
        return act
