###############################################################################
# Copyright (c) 2022 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Author: Narendra Chaudhary (Intel Corp.)                                       #
###############################################################################

import time
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math

import tpp_pytorch_extension
from distributed_xfold.xsmm_kernels.prototypes.Batched_DiffusionCrossAttention import BatchedCrossAttentionXSMM

torch.set_default_tensor_type(torch.FloatTensor)


class CrossAttention(nn.Module):
    """Multihead attention w/ Gating"""

    def __init__(self, num_head, a_dim, m_dim, output_dim):
        super().__init__()
        # self.config = config
        # self.global_config = global_config
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

    def forward(self, q_data, m_data, batched_bias):
        """Builds Attention module.
        Arguments:
          q_data: A tensor of queries, shape [batch_size, N_queries, q_channels].
          m_data: A tensor of memories from which the keys and values are
            projected, shape [batch_size, N_keys, m_channels].
          batched_bias: Shared bias, shape [batch_size, Heads, N_queries, N_keys].
        Returns:
          A float32 tensor of shape [batch_size, N_queries, output_dim].
        """

        # get query, key, value
        q = (torch.einsum("bqa,ahc->bqhc", q_data, self.query_w) + self.query_b) * self.key_dim ** (-0.5)
        k = torch.einsum("bka,ahc->bkhc", m_data, self.key_w)
        v = torch.einsum("bka,ahc->bkhc", m_data, self.value_w)

        logits = torch.einsum("bqhc,bkhc->bhqk", q, k) + batched_bias

        weights = self.softmax(logits)

        weighted_avg = torch.einsum("bhqk,bkhc->bqhc", weights, v)

        gate_values = (
            torch.einsum("bqc,chv->bqhv", q_data, self.gating_w)
        )
        gate_values = self.sigmoid(gate_values)
        weighted_avg *= gate_values
        return weighted_avg


set = 1
q_length = 128
k_length = 32
if set == 1:
    B, S, K, HS = 768, q_length, k_length, 256
    N, H = 8, 32

if set == 2:
    B, S, HS = q_length, q_length, 64
    N, H = 4, 16

if set == 3:
    B, S, HS = 320, q_length, 64
    N, H = 8, 8

if set == 4:
    B, S, HS = q_length, q_length, 128
    N, H = 4, 32

if set == 5:
    B, S, HS = q_length, 512, 256
    N, H = 8, 32


class Net1(nn.Module):  # First network containing original attention layer
    def __init__(self):
        super(Net1, self).__init__()
        self.attention = CrossAttention(
            num_head=N, a_dim=HS, m_dim=HS, output_dim=HS
        )  # Attention layer

    def forward(self, q_data, m_data, batched_bias):
        x = self.attention(q_data, m_data, batched_bias)
        return x


class Net2(nn.Module):  # Second network containing optimized attention layer
    def __init__(self):
        super(Net2, self).__init__()

        self.attention = BatchedCrossAttentionXSMM(
            num_head=N, a_dim=HS, m_dim=HS, output_dim=HS
        )  # Attention layer

    def forward(self, q_data, m_data, batched_bias):
        x = self.attention(q_data, m_data, batched_bias)
        return x


net1 = Net1()
net2 = Net2()

torch.manual_seed(1)  # Set random seed for reproducibility

q_data = torch.randn(B, S, HS, requires_grad=False)
m_data = torch.randn(B, K, HS, requires_grad=False)
batched_bias = torch.randn(B, N, S, K, requires_grad=False)
# nonbatched_bias = torch.Tensor()

query_w = torch.randn(HS, N, H)
query_b = torch.ones(1, N, H)
key_w = torch.randn(HS, N, H)
value_w = torch.randn(HS, N, H)
gating_w = torch.randn(HS, N, H)
gating_b = torch.randn(N, H)

net1.attention.query_w.data = query_w
net1.attention.query_b.data = query_b
net1.attention.key_w.data = key_w
net1.attention.value_w.data = value_w
net1.attention.gating_w.data = gating_w

net2.attention.query_w.data = query_w
net2.attention.query_b.data = query_b
net2.attention.key_w.data = key_w
net2.attention.value_w.data = value_w
net2.attention.gating_w.data = gating_w


Y1 = net1(q_data, m_data, batched_bias)
Y2 = net2(q_data, m_data, batched_bias)
r = Y1.max() - Y1.min()
print(Y1.shape)
print(Y2.shape)
print(
    "    Foward pass check: ",
    ((torch.abs(Y1 - Y2) / r < 0.0001).sum() == B * S * HS).item(),
)
# print("diff: ", r)
abs_error = torch.abs(Y1-Y2).sum()/torch.abs(Y1).sum()
print('abs error',(abs_error*100).item(),'%')
print(" Number of errors: ", B * S * HS - (torch.abs(Y1 - Y2) / r < 0.0001).sum())


forward1 = 0  # variables to store time values
forward2 = 0

N = 5  # Number of iterations

# with torch.profiler.profile(
#         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
#         on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/original_attention'),
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True,
#     ) as prof:
for _ in range(N):  # MKLDNN PyTorch layer Forward and Backward pass timing
    start = time.time()
    Y1 = net1(q_data, m_data, batched_bias)
    forward1 += time.time() - start
    # prof.step()

# tpp_pytorch_extension.reset_debug_timers()
# with torch.profiler.profile(
#         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
#         on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/optimized_attention'),
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True,
#     ) as prof:
for _ in range(N):  # Optimized PyTorch layer Forward and Backward pass timing
    start = time.time()
    Y2 = net2(q_data, m_data, batched_bias)
    forward2 += time.time() - start
    # prof.step()
# tpp_pytorch_extension.print_debug_timers()

print(
    "Forward pass time (PyTorch layer): {:.3f} ms | Forward pass time (Optimized layer): {:.3f} ms".format(
        forward1 * 1e3 / N, forward2 * 1e3 / N
    )
)
