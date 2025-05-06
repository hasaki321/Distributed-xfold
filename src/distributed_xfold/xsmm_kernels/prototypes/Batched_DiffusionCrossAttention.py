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


class BatchedCrossAttentionFunction(Function):
    @staticmethod
    def forward(
        ctx,
        q_data,
        m_data,
        batched_bias,
        query_w,
        query_b,
        key_w,
        value_w,
        gating_w,
        key_dim,
        value_dim,
    ):
        Bp_t, Sp_t = q_data.shape[:2]
        result = torch.ops._alpha_attention.batch_diffusion_cross_attention_forward(
            q_data.contiguous(),
            m_data.contiguous(),
            batched_bias.contiguous(),
            query_w,
            query_b,
            key_w,
            value_w,
            gating_w,
            key_dim,
            value_dim,
        )
        B_t, S_t, N_t, H_t = result.shape
        if S_t != Sp_t:
            result = result.narrow(1, 0, Sp_t)
        return result


def BatchedCrossAttentionXSMM_forward(
    self, q_data, m_data, batched_bias
):
    """Builds Attention module.
        Arguments:
          q_data: A tensor of queries, shape [batch_size, N_queries, q_channels].
          m_data: A tensor of memories from which the keys and values are
            projected, shape [batch_size, N_keys, m_channels].
          batched_bias: Shared bias, shape [batch_size, N_queries, N_keys, m_channels].
        Returns:
          A float32 tensor of shape [batch_size, N_queries, output_dim].
    """
    if (
        q_data.dtype == torch.bfloat16
    ):
        assert (m_data.dtype == torch.bfloat16) \
            and (batched_bias.dtype == torch.float32) \
            and (self.query_w.dtype == torch.bfloat16) \
            and (self.query_b.dtype == torch.float32) \
            and (self.key_w.dtype == torch.bfloat16) \
            and (self.value_w.dtype == torch.bfloat16) \
            and (self.gating_w.dtype == torch.bfloat16)

    output = BatchedCrossAttentionFunction.apply(
        q_data,
        m_data,
        batched_bias,
        self.query_w,
        self.query_b,
        self.key_w,
        self.value_w,
        self.gating_w,
        self.key_dim,
        self.value_dim,
    )
    
    
    return output


class BatchedCrossAttentionXSMM(nn.Module):
    """Multihead attention w/ Gating"""
    def __init__(self, num_head, a_dim, m_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim

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
        self.query_b = nn.Parameter(
            torch.Tensor(1, self.num_head, self.key_dim), requires_grad=False
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
        # softmax & act fn
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    # @torch.jit.ignore
    # def read_time(self) -> float:
    #     return time.time()

    def forward(self, q_data, m_data, batched_bias):
        """Builds Attention module.
        Arguments:
          q_data: A tensor of queries, shape [batch_size, N_queries, q_channels].
          m_data: A tensor of memories from which the keys and values are
            projected, shape [batch_size, N_keys, m_channels].
          batched_bias: Shared bias, shape [batch_size, N_queries, N_keys, m_channels].
        Returns:
          A float32 tensor of shape [batch_size, N_queries, output_dim].
        """

        if (q_data.dtype == torch.bfloat16):
            assert (m_data.dtype == torch.bfloat16) \
                and (batched_bias.dtype == torch.float32) \
                and (self.query_w.dtype == torch.bfloat16) \
                and (self.query_b.dtype == torch.float32) \
                and (self.key_w.dtype == torch.bfloat16) \
                and (self.value_w.dtype == torch.bfloat16) \
                and (self.gating_w.dtype == torch.bfloat16)


        output = BatchedCrossAttentionFunction.apply(
                q_data,
                m_data,
                batched_bias,
                self.query_w,
                self.query_b,
                self.key_w,
                self.value_w,
                self.gating_w,
                self.key_dim,
                self.value_dim,
        )
        return output
