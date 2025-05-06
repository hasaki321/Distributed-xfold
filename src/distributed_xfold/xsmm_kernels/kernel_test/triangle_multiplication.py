###############################################################################
# Copyright (c) 2023 Intel Corporation - All rights reserved.                 #
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
from distributed_xfold.xsmm_kernels.prototypes.TriangleMultiplication import TriangleMultiplicationXSMM
from xfold import fastnn 

torch.set_default_tensor_type(torch.FloatTensor)


class TriangleMultiplication(nn.Module):
    def __init__(self, c_pair: int = 128, equation: str = 'ckj,cki->cij') -> None:
        super(TriangleMultiplication, self).__init__()

        self.c_pair = c_pair

        self.left_norm_input = fastnn.LayerNorm(self.c_pair)
        self.projection = nn.Linear(self.c_pair, 2 * self.c_pair, bias=False)
        self.gate = nn.Linear(self.c_pair, 2 * self.c_pair, bias=False)
        self.center_norm = fastnn.LayerNorm(self.c_pair)
        self.output_projection = nn.Linear(
            self.c_pair, self.c_pair, bias=False)
        self.gating_linear = nn.Linear(self.c_pair, self.c_pair, bias=False)

        self.equation=equation
        # self.equation='ckj,cki->cij'
        # if _outgoing is True:
        #     self.equation='cik,cjk->cij'
        
    # receive a full copy of pair and mask at all rank
    def forward(self, pair: torch.Tensor, mask: torch.Tensor, head:int = 0) -> torch.Tensor:
        """
        Args:
            pair (torch.Tensor): [N_token // dp_size, N_token, c_pair]
            mask (torch.Tensor): [N_token // dp_size, N_token]
        Returns:
            torch.Tensor: [N_token // dp_size, N_token, c_pair]
        """
        pair = self.left_norm_input(pair)               # [N_token // dp_size, N_token, c_pair]
        input_pair = pair

        projection = self.projection(pair)              # [N_token // dp_size, N_token, c_pair // tp_size]
        projection = projection.permute(2, 0, 1)        # [c_pair // tp_size, N_token // dp_size, N_token]
        if mask is not None:
            projection *= mask[None, ...]

        gate = self.gate(pair)                          # [N_token // dp_size, N_token, 2 * c_pair // tp_size]
        gate = gate.permute(2, 0, 1)                    # [2 * c_pair // tp_size, N_token // dp_size, N_token]
        projection *= torch.sigmoid(gate)

        projection = projection.reshape(self.c_pair, 2, *projection.shape[1:])  # [c_pair, 2, N_token // dp_size, N_token]

        a, b = torch.chunk(projection, 2, dim=1)                          
        a, b = torch.squeeze(a, dim=1), torch.squeeze(b, dim=1)  # [c_pair // tp_size, N_token // dp_size, N_token]

        pair = torch.einsum(self.equation, a, b)     # [c_pair // tp_size , N_token // dp_size, N_token] | [c_pair // tp_size , N_token, N_token // dp_size]

        pair = pair.permute(1, 2, 0)                 # [N_token // dp_size, N_token, c_pair // tp_size]
        
        pair = self.center_norm(pair)                                   # [N_token // dp_size, N_token, c_pair]

        pair = self.output_projection(pair)                             # [N_token // dp_size, N_token, c_pair]
        gate_out = self.gating_linear(input_pair)                       # [N_token // dp_size, N_token, c_pair]

        pair *= torch.sigmoid(gate_out)

        return pair

set = 1
length = 764
if set == 1:
    B, S, HS = length, length, 64
    equation = "ikc,jkc->ijc"
    num_intermediate_channel = 64

if set == 2:
    B, S, HS = length, length, 128
    equation = "ikc,jkc->ijc"
    num_intermediate_channel = 128

if set == 3:
    B, S, HS = length, length, 64
    equation = "kjc,kic->ijc"
    num_intermediate_channel = 64

if set == 4:
    B, S, HS = length, length, 128
    equation = "kjc,kic->ijc"
    num_intermediate_channel = 128

equ_map = {
    "kjc,kic->ijc":'ckj,cki->cij',
    "ikc,jkc->ijc":'cik,cjk->cij',
}
class Net1(nn.Module):  # First network containing original attention layer
    def __init__(self):
        super(Net1, self).__init__()
        self.triangle_multiplication = TriangleMultiplication(
            equation=equ_map[equation],
            c_pair=HS,
        )  # Attention layer

    def forward(self, act, mask):
        y = self.triangle_multiplication(act,mask)
        return y


class Net2(nn.Module):  # Second network containing optimized attention layer
    def __init__(self):
        super(Net2, self).__init__()
        self.triangle_multiplication = TriangleMultiplicationXSMM(
            equation=equation,
            num_intermediate_channel=num_intermediate_channel,
            act_dim=HS,
        )  # Attention layer

    def forward(self, act, mask):
        x = self.triangle_multiplication(act, mask)
        return x


net1 = Net1()
net2 = Net2()

torch.manual_seed(11)  # Set random seed for reproducibility

act = torch.randn(B, S, HS, requires_grad=False)
mask = torch.rand(B, S, requires_grad=False)
net2.triangle_multiplication.left_norm_input.weight = torch.nn.Parameter(
    net1.triangle_multiplication.left_norm_input.weight
)
net2.triangle_multiplication.left_norm_input.bias = torch.nn.Parameter(
    net1.triangle_multiplication.left_norm_input.bias
)

t_proj = torch.cat(( net1.triangle_multiplication.projection.weight[::2, :],  net1.triangle_multiplication.projection.weight[1::2, :]),0)
net2.triangle_multiplication.projection.weight = torch.nn.Parameter(
    t_proj.contiguous()
)

t_gate = torch.cat(( net1.triangle_multiplication.gate.weight[::2, :],  net1.triangle_multiplication.gate.weight[1::2, :]),0)
net2.triangle_multiplication.gate.weight = torch.nn.Parameter(
    t_gate.contiguous()
)

net2.triangle_multiplication.center_norm.weight = torch.nn.Parameter(
    net1.triangle_multiplication.center_norm.weight
)
net2.triangle_multiplication.center_norm.bias = torch.nn.Parameter(
    net1.triangle_multiplication.center_norm.bias
)
net2.triangle_multiplication.output_projection.weight = torch.nn.Parameter(
    net1.triangle_multiplication.output_projection.weight
)
net2.triangle_multiplication.gating_linear.weight = torch.nn.Parameter(
    net1.triangle_multiplication.gating_linear.weight
)


# net2.triangle_multiplication.equation = equ_map[net2.triangle_multiplication.equation]


Y1 = net1(act, mask)
Y2 = net2(act, mask)

# print(Y1[1, 1, :10])
# print(Y2[1, 1, :10])
r = Y1.max() - Y1.min()
# print((torch.abs(Y1 - Y2) / r > 0.0001)[:, 0:4, :].sum())
print(
    "    Foward pass check: ",
    ((torch.abs(Y1 - Y2) / r < 0.0001).sum() == B * S * HS).item(),
)
# print("diff: ", r)
print(" Number of errors: ", B * S * HS - (torch.abs(Y1 - Y2) / r < 0.0001).sum())


forward1 = 0  # variables to store time values
forward2 = 0

N = 10  # Number of iterations

# with torch.profiler.profile(
#         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
#         on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/original_triangle_multiplication'),
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True,
#         with_flops=True
#     ) as prof:
for _ in range(N):  # MKLDNN PyTorch layer Forward and Backward pass timing
    start = time.time()
    Y1 = net1(act, mask)
    forward1 += time.time() - start
    # prof.step()

# tpp_pytorch_extension.reset_debug_timers()
# with torch.profiler.profile(
#         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
#         on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/long_triangle_multiplication'),
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True,
#         with_flops=True
#     ) as prof:
for _ in range(N):  # Optimized PyTorch layer Forward and Backward pass timing
    start = time.time()
    Y2 = net2(act, mask)
    forward2 += time.time() - start
    # prof.step()
# tpp_pytorch_extension.print_debug_timers()

print(
    "Forward pass time (PyTorch layer): {:.3f} ms | Forward pass time (Optimized layer): {:.3f} ms".format(
        forward1 * 1e3 / N, forward2 * 1e3 / N
    )
)
