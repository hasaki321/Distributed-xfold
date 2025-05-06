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
from distributed_xfold.xsmm_kernels.prototypes.Batched_DiffusionSelfAttention import BatchedSelfAttentionXSMM
import einops
torch.set_default_tensor_type(torch.FloatTensor)


class SelfAttention1(nn.Module):
# class GatingAttention(nn.Module):
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

    def forward(self, q_data, m_data, bias, nonbatched_bias=torch.Tensor()):
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

        # get query, key, value
        q = (torch.einsum("bqa,ahc->bqhc", q_data, self.query_w) + self.query_b) * self.key_dim ** (-0.5)
        k = torch.einsum("bka,ahc->bkhc", m_data, self.key_w)
        v = torch.einsum("bka,ahc->bkhc", m_data, self.value_w)

        logits = torch.einsum("bqhc,bkhc->bhqk", q, k) + bias

        if nonbatched_bias.shape[0] > 0:
            logits += torch.unsqueeze(nonbatched_bias, dim=0)
        weights = self.softmax(logits)

        weighted_avg = torch.einsum("bhqk,bkhc->bqhc", weights, v)

        gate_values = (
            torch.einsum("bqc,chv->bqhv", q_data, self.gating_w)
        )
        gate_values = self.sigmoid(gate_values)
        weighted_avg *= gate_values
        return weighted_avg

class SelfAttention(nn.Module):
    def __init__(self,
                 c_x: int = 768,
                 num_head: int = 16) -> None:

        super(SelfAttention, self).__init__()

        self.c_x = c_x
        self.num_head = num_head

        self.qkv_dim = self.c_x // self.num_head

        self.q_projection = nn.Linear(self.c_x, self.c_x, bias=True)
        self.k_projection = nn.Linear(self.c_x, self.c_x, bias=False)
        self.v_projection = nn.Linear(self.c_x, self.c_x, bias=False)

        self.gating_query = nn.Linear(self.c_x, self.c_x, bias=False)

    def forward(self,
                x1: torch.Tensor,
                x2: torch.Tensor,
                mask: torch.Tensor,
                pair_logits = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (num_tokens, ch)
            mask (torch.Tensor): (num_tokens,)
            pair_logits (torch.Tensor, optional): (num_heads, num_tokens, num_tokens)
        """

        q = self.q_projection(x1)
        k = self.k_projection(x2)
        v = self.v_projection(x2)

        q, k, v = map(lambda t: einops.rearrange(
            t, 'b n (h c) -> b h n c', h=self.num_head).unsqueeze(0), [q, k, v])

        scaling = q.size(-1) ** -0.5
        q = q * scaling
        logits = torch.matmul(q, k.transpose(-1, -2))

        if pair_logits is not None:
            logits += pair_logits

        if mask is not None:
            logits.masked_fill_(~(mask.bool()), -1e9)

        weights = torch.softmax(logits, dim=-1)

        weighted_avg = torch.matmul(weights, v)

        weighted_avg = weighted_avg.squeeze(0)
        weighted_avg = einops.rearrange(weighted_avg, 'b h q c -> b q (h c)')

        gate_logits = self.gating_query(x1)
        weighted_avg *= torch.sigmoid(gate_logits)
        return weighted_avg

set = 1
length = 764
if set == 1:
    B, S, HS = 512, length, 256
    N, H = 8, 32

if set == 2:
    B, S, HS = length, length, 64
    N, H = 4, 16

if set == 3:
    B, S, HS = 320, length, 64
    N, H = 8, 8

if set == 4:
    B, S, HS = length, length, 128
    N, H = 4, 32

if set == 5:
    B, S, HS = length, 512, 256
    N, H = 8, 32


class Net1(nn.Module):  # First network containing original attention layer
    def __init__(self):
        super(Net1, self).__init__()
        self.attention = SelfAttention(
            c_x = HS,
            num_head = N
        )  # Attention layer

    def forward(self, q_data, m_data, mask, nonbatched_bias):
        x = self.attention(q_data, m_data, mask, nonbatched_bias)
        return x


class Net2(nn.Module):  # Second network containing optimized attention layer
    def __init__(self):
        super(Net2, self).__init__()

        self.attention = BatchedSelfAttentionXSMM(
            num_head=N, a_dim=HS, m_dim=HS, output_dim=HS
        )  # Attention layer

    def forward(self, q_data, bias, nonbatched_bias):
        x = self.attention(q_data, bias, nonbatched_bias)
        return x

class Net3(nn.Module):  # Second network containing optimized attention layer
    def __init__(self):
        super(Net3, self).__init__()

        self.attention = SelfAttention1(
            num_head=N, a_dim=HS, m_dim=HS, output_dim=HS
        )  # Attention layer

    def forward(self, q_data, m_data, bias, nonbatched_bias):
        x = self.attention(q_data, m_data, bias, nonbatched_bias)
        return x

net1 = Net1()
net2 = Net2()
net3 = Net3()

net1 = torch.compile(net1)
net2 = torch.compile(net2)
net3 = torch.compile(net3)

torch.manual_seed(11)  # Set random seed for reproducibility

q_data = torch.randn(B, S, HS, requires_grad=False)
m_data = q_data.clone()
# m_data = torch.randn(B, S, HS, requires_grad=False)
bias = torch.randn(B, 1, 1, S, requires_grad=False)

p_true = 0.7 # 例如，希望大约 70% 的元素是 True (可以被 attend)
bias = (bias < p_true).float()

nonbatched_bias = torch.randn(N, S, S, requires_grad=False)
# nonbatched_bias = torch.Tensor()

query_w = torch.randn(HS, N, H)
query_b = torch.randn(1, N, H)
key_w = torch.randn(HS, N, H)
value_w = torch.randn(HS, N, H)
gating_w = torch.randn(HS, N, H)
# gating_b = torch.randn(N, H)

# net1.attention.query_w.data = query_w
# net1.attention.key_w.data = key_w
# net1.attention.value_w.data = value_w
# net1.attention.gating_w.data = gating_w

qkv_shape = (HS, N, H)
scale = H ** (-0.5)

net2.attention.query_w.data = net1.attention.q_projection.weight.T.reshape(qkv_shape).to(torch.bfloat16).contiguous()
net2.attention.query_b.data = net1.attention.q_projection.bias.reshape((1,) + qkv_shape[1:]).to(torch.float32).contiguous()
net2.attention.key_w.data = net1.attention.k_projection.weight.T.reshape(qkv_shape).to(torch.bfloat16).contiguous()
net2.attention.value_w.data = net1.attention.v_projection.weight.T.reshape(qkv_shape).to(torch.bfloat16).contiguous()
net2.attention.gating_w.data = net1.attention.gating_query.weight.T.reshape(qkv_shape).to(torch.bfloat16).contiguous()

net3.attention.query_w.data = net1.attention.q_projection.weight.T.reshape(qkv_shape)
net3.attention.query_b.data = net1.attention.q_projection.bias.reshape((1,) + qkv_shape[1:]).to(torch.float32).contiguous()
net3.attention.key_w.data = net1.attention.k_projection.weight.T.reshape(qkv_shape)
net3.attention.value_w.data = net1.attention.v_projection.weight.T.reshape(qkv_shape)
net3.attention.gating_w.data = net1.attention.gating_query.weight.T.reshape(qkv_shape)
# net2.attention.gating_b.data = gating_b


Y1 = net1(q_data, m_data, bias, nonbatched_bias)
Y2 = net2(q_data.to(torch.bfloat16).contiguous(), (1e9 * (bias-1)), nonbatched_bias).flatten(-2, -1)
Y3 = net3(q_data, m_data, (1e9 * (bias-1)), nonbatched_bias).flatten(-2, -1)
r = Y1.max() - Y1.min()
# print(Y1[0], Y2[0])
print(Y1.shape)
print(Y2.shape)
print(Y3.shape)

print(
    "    Foward pass check: ",
    ((torch.abs(Y1 - Y2) / r < 0.0001).sum() == B * S * HS).item(),
)
# print("diff: ", r)
abs_error = torch.abs(Y1-Y2).sum()/torch.abs(Y1).sum()
print('abs error',(abs_error*100).item(),'%')
print(" Number of errors: ", B * S * HS - (torch.abs(Y1 - Y2) / r < 0.0001).sum())

print(
    "    Foward pass check: ",
    ((torch.abs(Y1 - Y3) / r < 0.0001).sum() == B * S * HS).item(),
)
# print("diff: ", r)
abs_error = torch.abs(Y1-Y3).sum()/torch.abs(Y1).sum()
print('abs error',(abs_error*100).item(),'%')
print(" Number of errors: ", B * S * HS - (torch.abs(Y1 - Y3) / r < 0.0001).sum())

print(
    "    Foward pass check: ",
    ((torch.abs(Y3 - Y2) / (Y3.max() - Y3.min()) < 0.0001).sum() == B * S * HS).item(),
)
# print("diff: ", r)
abs_error = torch.abs(Y3-Y2).sum()/torch.abs(Y3).sum()
print('abs error',(abs_error*100).item(),'%')
print(" Number of errors: ", B * S * HS - (torch.abs(Y3 - Y2) / (Y3.max() - Y3.min()) < 0.0001).sum())

forward1 = 0  # variables to store time values
forward2 = 0

N = 10  # Number of iterations

# with torch.profiler.profile(
#         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
#         on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/original_attention'),
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True,
#     ) as prof:
for _ in range(N):  # MKLDNN PyTorch layer Forward and Backward pass timing
    start = time.time()
    Y1 = net1(q_data, m_data,  bias, nonbatched_bias)
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
    Y2 = net2(q_data.to(torch.bfloat16).contiguous(), bias, nonbatched_bias)
    forward2 += time.time() - start
    # prof.step()
# tpp_pytorch_extension.print_debug_timers()

print(
    "Forward pass time (PyTorch layer): {:.3f} ms | Forward pass time (Optimized layer): {:.3f} ms".format(
        forward1 * 1e3 / N, forward2 * 1e3 / N
    )
)
