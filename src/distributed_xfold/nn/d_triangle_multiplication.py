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
import torch.distributed as dist
from distributed_xfold.distribute_utils import all_gather_into_tensor, ShardInfo, shard_linear, init_dist_info
from distributed_xfold.xsmm_kernels.prototypes.TriangleMultiplication import TriangleMultiplicationXSMM_forward


class DistributeTriangleMultiplication(nn.Module):
    def __init__(self, device_mesh:torch.Tensor, c_pair: int = 128, _outgoing: bool = True) -> None:
        super(DistributeTriangleMultiplication, self).__init__()

        self.c_pair = c_pair

        self.left_norm_input = fastnn.LayerNorm(self.c_pair)
        self.projection = nn.Linear(self.c_pair, 2 * self.c_pair, bias=False)
        self.gate = nn.Linear(self.c_pair, 2 * self.c_pair, bias=False)
        self.center_norm = fastnn.LayerNorm(self.c_pair)
        self.output_projection = nn.Linear(
            self.c_pair, self.c_pair, bias=False)
        self.gating_linear = nn.Linear(self.c_pair, self.c_pair, bias=False)

        self._outgoing = _outgoing
        self.equation='ckj,cki->cij'
        if _outgoing is True:
            self.equation='cik,cjk->cij'

        init_dist_info(self, device_mesh)

    def shard_params(self): 
        self.projection = shard_linear(self.projection, self.rank, self.device_mesh, True)
        self.gate = shard_linear(self.gate, self.rank, self.device_mesh, True)
        
    # receive a full copy of pair and mask at all rank
    def forward(self, pair: torch.Tensor, mask: torch.Tensor, head:int = 0) -> torch.Tensor:
        """
        Args:
            pair (torch.Tensor): [N_token // dp_size, N_token, c_pair]
            mask (torch.Tensor): [N_token // dp_size, N_token]
        Returns:
            torch.Tensor: [N_token // dp_size, N_token, c_pair]
        """
        n_token = max(pair.shape[:-1])
        pair = self.left_norm_input(pair)               # [N_token // dp_size, N_token, c_pair]
        input_pair = pair

        projection = self.projection(pair)              # [N_token // dp_size, N_token, c_pair // tp_size]
        projection = projection.permute(2, 0, 1)        # [c_pair // tp_size, N_token // dp_size, N_token]
        if mask is not None:
            projection *= mask[None, ...]

        gate = self.gate(pair)                          # [N_token // dp_size, N_token, 2 * c_pair // tp_size]
        gate = gate.permute(2, 0, 1)                    # [2 * c_pair // tp_size, N_token // dp_size, N_token]
        projection *= torch.sigmoid(gate)

        projection = projection.reshape(self.c_pair // self.tp_size, 2, *projection.shape[1:])  # [c_pair, 2, N_token // dp_size, N_token]

        a, b = torch.chunk(projection, 2, dim=1)                          
        a, b = torch.squeeze(a, dim=1), torch.squeeze(b, dim=1)  # [c_pair // tp_size, N_token // dp_size, N_token]

        if self.use_dp:
            full_b = torch.zeros((self.c_pair // self.tp_size, n_token, n_token), dtype=b.dtype, device=b.device)
            all_gather_into_tensor(full_b, b, 
                                group=self.dp_shard_group, 
                                async_op=False,)
            b = full_b

        pair = torch.einsum(self.equation, a, b)     # [c_pair // tp_size , N_token // dp_size, N_token] | [c_pair // tp_size , N_token, N_token // dp_size]

        pair = pair.permute(1, 2, 0)                 # [N_token // dp_size, N_token, c_pair // tp_size]

        if self.use_tp:
            full_pair = torch.zeros(pair.shape[:-1] + (self.c_pair,), dtype=pair.dtype, device=pair.device)
            all_gather_into_tensor(full_pair, pair, 
                                group=self.tp_shard_group, 
                                async_op=False,)
            pair = full_pair
        
        pair = self.center_norm(pair)                                   # [N_token // dp_size, N_token, c_pair]

        pair = self.output_projection(pair)                             # [N_token // dp_size, N_token, c_pair]
        gate_out = self.gating_linear(input_pair)                       # [N_token // dp_size, N_token, c_pair]

        pair *= torch.sigmoid(gate_out)

        return pair



class XSMMTriangleMultiplication(nn.Module):
    def __init__(self, c_pair: int = 128, _outgoing: bool = True) -> None:
        super(XSMMTriangleMultiplication, self).__init__()

        self.c_pair = c_pair

        self.left_norm_input = fastnn.LayerNorm(self.c_pair)
        self.projection = nn.Linear(self.c_pair, 2 * self.c_pair, bias=False)
        self.gate = nn.Linear(self.c_pair, 2 * self.c_pair, bias=False)
        self.center_norm = fastnn.LayerNorm(self.c_pair)
        self.output_projection = nn.Linear(
            self.c_pair, self.c_pair, bias=False)
        self.gating_linear = nn.Linear(self.c_pair, self.c_pair, bias=False)

        self._outgoing = _outgoing
        self.c_equation="kjc,kic->ijc"
        if _outgoing is True:
            self.c_equation="ikc,jkc->ijc"

        self.equation='ckj,cki->cij'
        if _outgoing is True:
            self.equation='cik,cjk->cij'

    def load_xsmm_params(self):
        dtype = torch.bfloat16
        t_proj = torch.cat((self.projection.weight[::2, :], self.projection.weight[1::2, :]),0)
        t_gate = torch.cat((self.gate.weight[::2, :], self.gate.weight[1::2, :]),0)
        self.left_norm_input.weight = torch.nn.Parameter(self.left_norm_input.weight.to(dtype).contiguous(), requires_grad=False)
        self.left_norm_input.bias = torch.nn.Parameter(self.left_norm_input.bias.to(dtype).contiguous(), requires_grad=False)
        self.center_norm.weight = torch.nn.Parameter(self.center_norm.weight.to(dtype).contiguous(), requires_grad=False)
        self.center_norm.bias = torch.nn.Parameter(self.center_norm.bias.to(dtype).contiguous(), requires_grad=False)

        self.projection.weight = nn.Parameter(t_proj.to(dtype).contiguous(), requires_grad=False)
        self.gate.weight = nn.Parameter(t_gate.to(dtype).contiguous(), requires_grad=False)
        self.output_projection.weight = nn.Parameter(self.output_projection.weight.to(dtype).contiguous(), requires_grad=False)
        self.gating_linear.weight = nn.Parameter(self.gating_linear.weight.to(dtype).contiguous(), requires_grad=False)

    # receive a full copy of pair and mask at all rank
    def forward(self, pair: torch.Tensor, mask: torch.Tensor, head:int = 0) -> torch.Tensor:
        """
        Args:
            pair (torch.Tensor): [N_token // dp_size, N_token, c_pair]
            mask (torch.Tensor): [N_token // dp_size, N_token]
        Returns:
            torch.Tensor: [N_token // dp_size, N_token, c_pair]
        """
        n_token = max(pair.shape[:-1])
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

    def xsmm_forward(self, pair: torch.Tensor, mask: torch.Tensor, head:int = 0) -> torch.Tensor:
        dtype = pair.dtype
        # is_16bit = dtype in [torch.float16, torch.bfloat16]
        # if is_16bit:
        #     pair = pair.float().contiguous() # Upcast to float32
        #     mask = mask.float() # Upcast to float32

        pair = TriangleMultiplicationXSMM_forward(self, pair.to(torch.bfloat16), mask)
        
        # if is_16bit:
        pair = pair.to(dtype) # Cast back to original dtype
        return pair
        
