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
from distributed_xfold.xsmm_kernels.kernel_test.triangle_multiplication import TriangleMultiplication

torch.set_default_tensor_type(torch.FloatTensor)



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
    net1.triangle_multiplication.left_norm_input.weight.to(torch.bfloat16).contiguous()
)
net2.triangle_multiplication.left_norm_input.bias = torch.nn.Parameter(
    net1.triangle_multiplication.left_norm_input.bias.to(torch.bfloat16).contiguous()
)

t_proj = torch.cat(( net1.triangle_multiplication.projection.weight[::2, :],  net1.triangle_multiplication.projection.weight[1::2, :]),0).to(torch.bfloat16)
net2.triangle_multiplication.projection.weight = torch.nn.Parameter(
    t_proj.contiguous()
)

t_gate = torch.cat(( net1.triangle_multiplication.gate.weight[::2, :],  net1.triangle_multiplication.gate.weight[1::2, :]),0).to(torch.bfloat16)
net2.triangle_multiplication.gate.weight = torch.nn.Parameter(
    t_gate.contiguous()
)

net2.triangle_multiplication.center_norm.weight = torch.nn.Parameter(
    net1.triangle_multiplication.center_norm.weight.to(torch.bfloat16).contiguous()
)
net2.triangle_multiplication.center_norm.bias = torch.nn.Parameter(
    net1.triangle_multiplication.center_norm.bias.to(torch.bfloat16).contiguous()
)
net2.triangle_multiplication.output_projection.weight = torch.nn.Parameter(
    net1.triangle_multiplication.output_projection.weight.to(torch.bfloat16).contiguous()
)
net2.triangle_multiplication.gating_linear.weight = torch.nn.Parameter(
    net1.triangle_multiplication.gating_linear.weight.to(torch.bfloat16).contiguous()
)

Y1 = net1(act, mask)
Y2 = net2(act.to(torch.bfloat16), mask.to(torch.bfloat16))

# print(Y1[1, 1, :10])
# print(Y2[1, 1, :10])
r = Y1.max() - Y1.min()
# print((torch.abs(Y1 - Y2) / r > 0.1)[:, :, :].sum())
print(
    "    Foward pass check: ",
    ((torch.abs(Y1 - Y2.type(torch.float32)) / r < 0.1).sum() == B * S * HS).item(),
)
# print("diff: ", r)
print(
    " Number of errors: ",
    B * S * HS - (torch.abs(Y1 - Y2.type(torch.float32)) / r < 0.1).sum(),
)

abs_error = torch.abs(Y1-Y2).sum()/torch.abs(Y1).sum()
print('abs error',(abs_error*100).item(),'%')

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
    Y2 = net2(act.to(torch.bfloat16), mask.to(torch.bfloat16))
    forward2 += time.time() - start
    # prof.step()
# tpp_pytorch_extension.print_debug_timers()

print(
    "Forward pass time (PyTorch layer): {:.3f} ms | Forward pass time (Optimized layer): {:.3f} ms".format(
        forward1 * 1e3 / N, forward2 * 1e3 / N
    )
)
