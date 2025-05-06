###############################################################################
# Copyright (c) 2022 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Author: Narendra Chaudhary (Intel Corp.)                                       #
###############################################################################

import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.autograd import Function

import time
from contextlib import contextmanager


class TransitionFunction(Function):
    @staticmethod
    def forward(
        ctx,
        act,
        transition1,
        transition2,
        layernorm_weight,
        layernorm_bias
    ):
        result = torch.ops._alpha_attention.transition_forward(
            act.contiguous(),
            transition1,
            transition2,
            layernorm_weight,
            layernorm_bias
        )
        return result


def TransitionXSMM_forward(self, act):
    """Builds Attention module.
    Arguments:
      act: A tensor of queries, shape [batch_size, N_queries, q_channels].
    """
    if (act.dtype == torch.bfloat16):
        assert self.transition1.dtype == torch.bfloat16 \
            and self.transition2.dtype == torch.float32 \
            and self.input_layer_norm.weight.dtype == torch.bfloat16 \
            and self.input_layer_norm.bias.dtype == torch.bfloat16

    output = TransitionFunction.apply(
        act.contiguous(),
        self.transition1,
        self.transition2,
        self.input_layer_norm.weight,
        self.input_layer_norm.bias
    )
    return output


class TransitionXSMM(nn.Module):
    """Multihead attention w/ Gating"""
    def __init__(self, c_x: int, num_intermediate_factor: int = 4) -> None:
        super(TransitionXSMM, self).__init__()
        self.num_intermediate_factor = num_intermediate_factor
        self.c_in = c_x
        self.input_layer_norm = torch.nn.LayerNorm((c_x,))
        self.transition1 = nn.Parameter(
            torch.Tensor(c_x, self.num_intermediate_factor * c_x * 2), requires_grad=False
        )
        self.transition2 = nn.Parameter(
            torch.Tensor(self.num_intermediate_factor * c_x, c_x), requires_grad=False
        )

    def forward(self, act):
        """Builds Attention module.
        Arguments:
        act: A tensor of queries, shape [batch_size, N_queries, q_channels].
        """
        if (act.dtype == torch.bfloat16):
            assert self.transition1.dtype == torch.bfloat16 \
                and self.transition2.dtype == torch.float32 \
                and self.input_layer_norm.weight.dtype == torch.bfloat16 \
                and self.input_layer_norm.bias.dtype == torch.bfloat16

        output = TransitionFunction.apply(
            act.contiguous(),
            self.transition1,
            self.transition2,
            self.input_layer_norm.weight,
            self.input_layer_norm.bias
        )
        return output
