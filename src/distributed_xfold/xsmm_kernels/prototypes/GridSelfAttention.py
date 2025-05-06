# Distributed xfold authors
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
from torch.autograd import Function


class GridSelfAttentionFunction(Function):
    @staticmethod
    def forward(
        ctx,
        q_data,
        # m_data,
        bias,
        nonbatched_bias,
        query_w,
        key_w,
        value_w,
        gating_w,
        output_w,
        key_dim,
        value_dim,
    ):
        Bp_t, Sp_t = q_data.shape[:2]
        result = torch.ops._alpha_attention.grid_self_attn_forward(
            q_data.contiguous(),
            bias.contiguous(),
            nonbatched_bias.contiguous(),
            query_w,
            key_w,
            value_w,
            gating_w,
            output_w,
            key_dim,
            value_dim,
        )
        B_t, S_t, Hs_t = result.shape
        if S_t != Sp_t:
            result = result.narrow(1, 0, Sp_t)
        return result
    
def GridSelfAttentionXSMM_forward(
    self, q_data, bias, nonbatched_bias=torch.Tensor()
):
    """Builds Attention module.
    Arguments:
      q_data: A tensor of queries, shape [batch_size, N_queries, q_channels].
      m_data: A tensor of memories from which the keys and values are
        projected, shape [batch_size, N_keys, m_channels].
      bias: A bias for the attention, shape [batch_size, N_queries, N_keys].
      nonbatched_bias: Shared bias, shape [N_queries, N_keys].
    Returns:
      A float32 tensor of shape [batch_size, N_queries, output_dim].
    """
    
    if (q_data.dtype == torch.bfloat16):
            assert self.query_w.dtype == torch.bfloat16 \
                and self.key_w.dtype == torch.bfloat16 \
                and self.value_w.dtype == torch.bfloat16 \
                and self.gating_w.dtype == torch.bfloat16 \
                and self.output_w.dtype == torch.bfloat16 \
                
    output = GridSelfAttentionFunction.apply(
        q_data,
        bias.float(),
        nonbatched_bias.float(),
        self.query_w,
        self.key_w,
        self.value_w,
        self.gating_w,
        self.output_w,
        self.key_dim,
        self.value_dim,
    )

    return output


class GrigSelfAttentionXSMM(nn.Module):
    """Multihead attention w/ Gating"""

    # def __init__(self, config, global_config, a_dim, m_dim, output_dim):
    def __init__(self, num_head, a_dim, m_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        # k,v dim
        self.key_dim = int(a_dim)
        self.value_dim = int(m_dim)
        self.num_head = num_head
        assert self.key_dim % self.num_head == 0
        assert self.value_dim % self.num_head == 0
        self.key_dim = self.key_dim // self.num_head
        self.value_dim = self.value_dim // self.num_head

        # q,k,v weights
        self.query_w = nn.Parameter(
            torch.Tensor(a_dim, self.num_head, self.key_dim), requires_grad=False
        )
        self.key_w = nn.Parameter(
            torch.Tensor(m_dim, self.num_head, self.key_dim), requires_grad=False
        )
        self.value_w = nn.Parameter(
            torch.Tensor(m_dim, self.num_head, self.value_dim), requires_grad=False
        )
        self.gating_w = nn.Parameter(
            torch.Tensor(a_dim, self.num_head, self.value_dim), requires_grad=False
        )
        self.output_w = nn.Parameter(
            torch.Tensor(self.num_head, self.value_dim, self.output_dim),
            requires_grad=False,
        )
        # softmax & act fn
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    # @torch.jit.ignore
    # def read_time(self) -> float:
    #     return time.time()

    def forward(self, q_data, bias, nonbatched_bias=torch.Tensor()):
        """Builds Attention module.
        Arguments:
          q_data: A tensor of queries, shape [batch_size, N_queries, q_channels].
          m_data: A tensor of memories from which the keys and values are
            projected, shape [batch_size, N_keys, m_channels].
          bias: A bias for the attention, shape [batch_size, N_queries, N_keys].
          nonbatched_bias: Shared bias, shape [N_queries, N_keys].
        Returns:
          A float32 tensor of shape [batch_size, N_queries, output_dim].
        """

        if (q_data.dtype == torch.bfloat16):
            assert self.query_w.dtype == torch.bfloat16 \
                and self.key_w.dtype == torch.bfloat16 \
                and self.value_w.dtype == torch.bfloat16 \
                and self.gating_w.dtype == torch.bfloat16 \
                and self.output_w.dtype == torch.bfloat16 \
                
        output = GridSelfAttentionFunction.apply(
            q_data,
            bias.float(),
            nonbatched_bias.float(),
            self.query_w,
            self.key_w,
            self.value_w,
            self.gating_w,
            self.output_w,
            self.key_dim,
            self.value_dim,
        )

        return output
