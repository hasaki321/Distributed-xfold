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
from distributed_xfold.xsmm_kernels.prototypes.Transition import TransitionXSMM

torch.set_default_tensor_type(torch.FloatTensor)


class Transition(nn.Module):

    def __init__(self, c_x: int, num_intermediate_factor: int = 4) -> None:
        super(Transition, self).__init__()
        self.num_intermediate_factor = num_intermediate_factor
        self.c_in = c_x
        self.input_layer_norm = torch.nn.LayerNorm((c_x,))
        self.transition1 = nn.Parameter(
            torch.Tensor(c_x, self.num_intermediate_factor * c_x * 2), requires_grad=False
        )
        self.transition2 = nn.Parameter(
            torch.Tensor(self.num_intermediate_factor * c_x, c_x), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer_norm(x)
        y = torch.matmul(x, self.transition1)
        a, b = torch.chunk(y, 2, dim=-1)
        c = torch.nn.functional.silu(a) * b
        return torch.matmul(c, self.transition2)

set = 1
length = 768
if set == 1:
    B, S, HS = 768, length, 64

if set == 2:
    B, S, HS = length, length, 64

if set == 3:
    B, S, HS = 320, length, 64

if set == 4:
    B, S, HS = length, length, 128

if set == 5:
    B, S, HS = length, 512, 256


class Net1(nn.Module):  # First network containing original attention layer
    def __init__(self):
        super(Net1, self).__init__()
        self.transition = Transition(
            c_x=HS, num_intermediate_factor=4
        )

    def forward(self, x):
        return self.transition(x)


class Net2(nn.Module):  # Second network containing optimized attention layer
    def __init__(self):
        super(Net2, self).__init__()
        self.transition = TransitionXSMM(
            c_x=HS, num_intermediate_factor=4
        )

    def forward(self, x):
        return self.transition(x)


net1 = Net1()
net2 = Net2()

torch.manual_seed(11)  # Set random seed for reproducibility

x = torch.randn(B, S, HS, requires_grad=False)

net2.transition.input_layer_norm.weight.data = net1.transition.input_layer_norm.weight.data
net2.transition.input_layer_norm.bias.data = net1.transition.input_layer_norm.bias.data
net2.transition.transition1.data = net1.transition.transition1.data
net2.transition.transition2.data = net1.transition.transition2.data

Y1 = net1(x)
print('pass 1')
Y2 = net2(x)
print('pass 2')
r = Y1.max() - Y1.min()

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
    Y1 = net1(x)
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
    Y2 = net2(x)
    forward2 += time.time() - start
    # prof.step()
# tpp_pytorch_extension.print_debug_timers()

print(
    "Forward pass time (PyTorch layer): {:.3f} ms | Forward pass time (Optimized layer): {:.3f} ms".format(
        forward1 * 1e3 / N, forward2 * 1e3 / N
    )
)
