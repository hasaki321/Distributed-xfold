from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from xfold.nn import atom_layout
from xfold import fastnn
from xfold.nn.diffusion_transformer import (
    AdaptiveLayerNorm, 
    AdaLNZero, 
    DiffusionTransition, 
    SelfAttention,
    DiffusionCrossAttTransformer,
    CrossAttention,
    DiffusionTransformer
)
from distributed_xfold.xsmm_kernels.prototypes.Batched_DiffusionSelfAttention import BatchedSelfAttentionXSMM_forward
from distributed_xfold.nn.d_diffusion_transformer import DistributeAdaLNZero

import torch.distributed as dist
from distributed_xfold.distribute_utils import shard_linear, DeviceMesh

class DistributeAdaLNZero(AdaLNZero):
    def __init__(self,
                 c_in: int,
                 c_out: int,
                 c_single_cond: int,
                 use_single_cond: bool = False) -> None:
        super().__init__(c_in,c_out,c_single_cond,use_single_cond)
        self.tpp_transition2 = None 

    def forward(self,
                x: torch.Tensor,
                single_cond: Optional[torch.Tensor] = None) -> torch.Tensor:

        assert (single_cond is None) == (self.use_single_cond is False)

        if self.tpp_transition2 is not None:
            output = torch.einsum('...hk, hkc -> ...c', x, self.tpp_transition2)
        else:
            output = self.transition2(x)

        return output

class SelfAttention1(nn.Module):
    """Multihead attention w/ Gating"""

    def __init__(self, num_head=16, a_dim=768, m_dim=768):
        super().__init__()
        self.key_dim = int(a_dim)
        self.value_dim = int(m_dim)
        self.num_head = num_head
        print(self.key_dim, self.num_head)
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

    def forward(self, q_data, m_data, mask, nonbatched_bias):
        """Builds Attention module.
        Arguments:
          q_data: A tensor of queries, shape [batch_size, N_queries, q_channels].
          m_data: A tensor of memories from which the keys and values are
            projected, shape [batch_size, N_keys, m_channels].
          mask: A mask for the attention, shape [N_queries, N_keys].
          nonbatched_bias: Shared bias, shape [N_queries, N_keys].
        Returns:
          A float32 tensor of shape [batch_size, N_queries, output_dim].
        """

        # get query, key, value
        q_data = self.adaptive_layernorm(q_data, None)
        m_data = self.adaptive_layernorm(m_data, None)
        print(q_data.shape, self.query_w.shape, self.query_b.shape)
        print(m_data.shape, self.key_w.shape)
        q = (torch.einsum("bqa,ahc->bqhc", q_data, self.query_w) + self.query_b) * self.key_dim ** (-0.5)
        k = torch.einsum("bka,ahc->bkhc", m_data, self.key_w)
        v = torch.einsum("bka,ahc->bkhc", m_data, self.value_w)
        return q, k, v
        logits = torch.einsum("bqhc,bkhc->bhqk", q, k)

        if nonbatched_bias.shape[0] > 0:
            logits += torch.unsqueeze(nonbatched_bias, dim=0)

        mask = mask[None, None, :, :].to(dtype=torch.bool)
        logits.masked_fill_(~mask, -1e9)

        weights = self.softmax(logits)

        weighted_avg = torch.einsum("bhqk,bkhc->bqhc", weights, v)
        return weighted_avg
        gate_values = (
            torch.einsum("bqc,chv->bqhv", q_data, self.gating_w)
        )
        gate_values = self.sigmoid(gate_values)
        weighted_avg *= gate_values

        # output = weighted_avg.reshape(weighted_avg.shape[0],weighted_avg.shape[1],-1)
        return weighted_avg


class SelfAttention(nn.Module):
    def __init__(self,
                 c_x: int = 768,
                 c_single_cond: int = 384,
                 num_head: int = 16,
                 use_single_cond: bool = False) -> None:

        super(SelfAttention, self).__init__()

        self.c_x = c_x
        self.c_single_cond = c_single_cond
        self.num_head = num_head

        self.qkv_dim = self.c_x // self.num_head
        self.use_single_cond = use_single_cond

        self.adaptive_layernorm = AdaptiveLayerNorm(
            self.c_x, self.c_single_cond, self.use_single_cond)

        self.q_projection = nn.Linear(self.c_x, self.c_x, bias=True)
        self.k_projection = nn.Linear(self.c_x, self.c_x, bias=False)
        self.v_projection = nn.Linear(self.c_x, self.c_x, bias=False)

        self.gating_query = nn.Linear(self.c_x, self.c_x, bias=False)

        self.adaptive_zero_init = AdaLNZero(
            self.c_x, self.c_x, self.c_single_cond, self.use_single_cond)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor,
                pair_logits: Optional[torch.Tensor] = None,
                single_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass

class DistributeSelfAttention(SelfAttention):
    def __init__(self,
                 c_x: int = 768,
                 c_single_cond: int = 384,
                 num_head: int = 16,
                 use_single_cond: bool = False,
                 use_batch_infer: bool = False) -> None:

        super().__init__(c_x, c_single_cond, num_head, use_single_cond)

        self.adaptive_zero_init = DistributeAdaLNZero(
            self.c_x, self.c_x, self.c_single_cond, self.use_single_cond)
        
        self.use_batch_infer = use_batch_infer

        self.use_tp = False
        self.tp_size = 1

    def load_xsmm_params_(self): 
        dtype = torch.float32
        qkv_shape = (self.c_x,) + (self.num_head // self.tp_size, self.qkv_dim)

        self.query_w = nn.Parameter(self.q_projection.weight.T.reshape(qkv_shape)\
            .to(dtype).contiguous(), requires_grad=False)
        self.query_b = nn.Parameter(self.q_projection.bias.reshape(qkv_shape[1:])\
            .to(dtype).contiguous(), requires_grad=False)

        self.key_w = nn.Parameter(self.k_projection.weight.T.reshape(qkv_shape)\
            .to(dtype).contiguous(), requires_grad=False)
        self.value_w = nn.Parameter(self.v_projection.weight.T.reshape(qkv_shape)\
            .to(dtype).contiguous(), requires_grad=False)

        self.gating_w = nn.Parameter(self.gating_query.weight.T.reshape(qkv_shape)\
            .to(dtype).contiguous(), requires_grad=False
        )

        self.adaptive_zero_init.tpp_transition2 = nn.Parameter(self.adaptive_zero_init.transition2.weight.T.\
            reshape((self.num_head // self.tp_size, self.qkv_dim, self.c_x)).to(dtype).contiguous(), requires_grad=False
        )

        self.key_dim = self.value_dim = self.qkv_dim

    def xsmm_forward(self,
                x: torch.Tensor,
                mask: torch.Tensor,
                pair_logits: Optional[torch.Tensor] = None,
                single_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert (single_cond is None) == (self.use_single_cond is False)

        x = self.adaptive_layernorm(x, single_cond)

        nonbatched_bias = 1e9 * (mask[None, ...].float() - 1)
        if pair_logits is not None:
            nonbatched_bias = nonbatched_bias + pair_logits

        dtype = x.dtype
        is_16bit = dtype in [torch.float16, torch.bfloat16] # Correct dtype check for PyTorch

        if is_16bit:
            x = x.float().contiguous() # Upcast to float32
            nonbatched_bias = nonbatched_bias.float().contiguous() # Upcast to float32

        if self.use_batch_infer:
            weighted_avg = BatchedSelfAttentionXSMM_forward(self, x, x, nonbatched_bias)
        else:
            x = x.unsqueeze(0)
            weighted_avg = BatchedSelfAttentionXSMM_forward(self, x, x, nonbatched_bias).squeeze(0)

        if is_16bit:
            weighted_avg = weighted_avg.to(dtype) # Cast back to original dtype

        return weighted_avg, self.adaptive_zero_init(weighted_avg, single_cond)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor,
                pair_logits: Optional[torch.Tensor] = None,
                single_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (num_tokens, ch)
            mask (torch.Tensor): (num_tokens,)
            pair_logits (torch.Tensor, optional): (num_heads, num_tokens, num_tokens)
        """

        assert (single_cond is None) == (self.use_single_cond is False)

        x = self.adaptive_layernorm(x, single_cond)
        # assert ~torch.isnan(x).any()

        q = self.q_projection(x)
        k = self.k_projection(x)
        v = self.v_projection(x)

        if self.use_batch_infer:
            q, k, v = map(lambda t: einops.rearrange(
                t, 'b n (h c) -> b h n c', h=self.num_head // self.tp_size), [q, k, v])
        else:
            q, k, v = map(lambda t: einops.rearrange(
                t, 'n (h c) -> h n c', h=self.num_head // self.tp_size).unsqueeze(0), [q, k, v])
        # assert ~torch.isnan(q).any() and ~torch.isnan(k).any() and ~torch.isnan(v).any()
        return q, k, v
        print(q.shape, mask.shape, pair_logits.shape)

        def dot_product_attention_torch(q: torch.Tensor,
                                k: torch.Tensor,
                                v: torch.Tensor,
                                mask: Optional[torch.Tensor] = None,
                                bias: Optional[torch.Tensor] = None):
            scaling = q.size(-1) ** -0.5
            q = q * scaling
            logits = torch.matmul(q, k.transpose(-1, -2))

            if bias is not None:
                logits += bias

            if mask is not None:
                logits.masked_fill_(~mask, -1e9)

            weights = torch.softmax(logits, dim=-1)

            return torch.matmul(weights, v)
        weighted_avg = dot_product_attention_torch(
            q, k, v, mask=mask[None, None, :, :], bias=pair_logits
        )
        print(weighted_avg.shape)
        return weighted_avg
        # assert ~torch.isnan(weighted_avg).any()

        if self.use_batch_infer:
            weighted_avg = einops.rearrange(weighted_avg, 'b h q c -> b q (h c)')
        else:
            weighted_avg = weighted_avg.squeeze(0)
            weighted_avg = einops.rearrange(weighted_avg, 'h q c -> q (h c)')

        gate_logits = self.gating_query(x)
        weighted_avg *= torch.sigmoid(gate_logits)
        # assert ~torch.isnan(weighted_avg).any()

        return weighted_avg, self.adaptive_zero_init(weighted_avg, single_cond)
    

if __name__ == "__main__":
    b = 16
    N_token = 32
    c_x: int = 768
    c_single_cond: int = 384
    num_head: int = 16
    with_single = False
    use_batch_infer = True
    dtype = torch.float32
    single = None
    
    x = torch.randn((b, N_token, c_x))
    mask = torch.randn((N_token,N_token)) > 0.7
    pair = torch.randn((num_head, N_token,N_token))

    self_attn = DistributeSelfAttention(use_single_cond=with_single, use_batch_infer=use_batch_infer)
    print('1', num_head, c_x)
    self_attn1 = SelfAttention1()
    opt_self_attn = DistributeSelfAttention(use_single_cond=with_single, use_batch_infer=use_batch_infer)
    opt_self_attn.load_state_dict(self_attn.state_dict())
    opt_self_attn.load_xsmm_params_()
    opt_self_attn.forward = opt_self_attn.xsmm_forward

    qkv_shape = (c_x,) + (num_head, c_x // num_head)

    self_attn1.adaptive_layernorm = self_attn.adaptive_layernorm
    self_attn1.query_w = nn.Parameter(self_attn.q_projection.weight.T.contiguous().reshape(qkv_shape)\
        .to(dtype).contiguous(), requires_grad=False)
    self_attn1.query_b = nn.Parameter(self_attn.q_projection.bias.contiguous().reshape(qkv_shape[1:])\
        .to(dtype).contiguous() * ((c_x // num_head) ** (-0.5)), requires_grad=False)

    self_attn1.key_w = nn.Parameter(self_attn.k_projection.weight.T.contiguous().reshape(qkv_shape)\
        .to(dtype).contiguous(), requires_grad=False)
    self_attn1.value_w = nn.Parameter(self_attn.v_projection.weight.T.contiguous().reshape(qkv_shape)\
        .to(dtype).contiguous(), requires_grad=False)

    self_attn1.gating_w = nn.Parameter(self_attn.gating_query.weight.T.contiguous().reshape(qkv_shape)\
        .to(dtype).contiguous(), requires_grad=False
    )


    # ==============================================
    torch.manual_seed(11)
    x = torch.randn((b , N_token, c_x))
    x = torch.nn.functional.layer_norm(x, (c_x,))

    mask = torch.randn((N_token,N_token)) > 0.7
    pair_logits = torch.randn((num_head , N_token, N_token))

    scale = (c_x // num_head) ** (-0.5)
    
    q_w_1 = torch.randn((c_x, c_x))
    q_b_1 = torch.randn((c_x,))

    k_w_1 = torch.randn((c_x, c_x))
    v_w_1 = torch.randn((c_x, c_x))

    g_w_1 = torch.randn((c_x, c_x))

    q_w_2 = q_w_1.T.reshape(qkv_shape)
    q_b_2 = q_b_1.reshape(qkv_shape[1:]) * scale

    k_w_2 = k_w_1.T.reshape(qkv_shape)
    v_w_2 = v_w_1.T.reshape(qkv_shape)

    g_w_2 = g_w_1.T.reshape(qkv_shape)
    

    q1 = x @ q_w_1.T + q_b_1
    q1 = einops.rearrange(q1, 'b n (h c) -> b h n c', h=num_head) * scale

    q2 = torch.einsum("bqa,ahc->bqhc", x, q_w_2) * scale + q_b_2
    assert torch.allclose(q1, q2.transpose(1,2), rtol=1e-4, atol=1e-5)

    k1 = x @ k_w_1.T
    k1 = einops.rearrange(k1, 'b n (h c) -> b h n c', h=num_head)

    k2 = torch.einsum("bqa,ahc->bqhc", x, k_w_2)
    assert torch.allclose(k1, k2.transpose(1,2))

    v1 = x @ v_w_1.T
    v1 = einops.rearrange(v1, 'b n (h c) -> b h n c', h=num_head)

    v2 = torch.einsum("bqa,ahc->bqhc", x, v_w_2)
    assert torch.allclose(v1, v2.transpose(1,2))

    # =======================1
    logits1 = torch.matmul(q1, k1.transpose(-1, -2))

    if pair_logits is not None:
        logits1 += pair_logits

    logits1.masked_fill_(~mask[None, None, :, :].to(dtype=torch.bool), -1e9)

    weights1 = torch.softmax(logits1, dim=-1)

    # =======================2

    logits2 = torch.einsum("bqhc,bkhc->bhqk", q2, k2)

    if pair_logits.shape[0] > 0:
        logits2 += torch.unsqueeze(pair_logits, dim=0)

    # logits2.masked_fill_(~mask[None, None, :, :].to(dtype=torch.bool), -1e9)
    logits2 += (mask[None, None, :, :].float() - 1) * 1e9

    weights2 = torch.softmax(logits2, dim=-1)

    # print(weights1[0], weights2[0])
    assert torch.allclose(weights1, weights2, rtol=1e-2, atol=1e-5)

    # =======================
    weighted_avg1 = torch.matmul(weights1, v1)
    weighted_avg2 = torch.einsum("bhqk,bkhc->bqhc", weights2, v2)
    assert torch.allclose(weighted_avg1, weighted_avg2.transpose(1, 2), rtol=1e-2, atol=1e-3)

    # =======================1

    weighted_avg1 = einops.rearrange(weighted_avg1, 'b h q c -> b q (h c)')
    gate_logits1 = x @ g_w_1.T
    weighted_avg1 *= torch.sigmoid(gate_logits1)

    # =======================2
    gate_logits2 = (
        torch.einsum("bqc,chv->bqhv", x, g_w_2)
    )
    weighted_avg2 *= torch.sigmoid(gate_logits2)
    assert torch.allclose(weighted_avg1, weighted_avg2.flatten(-2, -1), rtol=1e-2, atol=1e-3)


    # out1 = self_attn(x, mask, pair, single)[0]
    # out2 = self_attn1(x, x.clone(), mask, pair)[0].transpose(1,2)
    # out3 = opt_self_attn(x, mask, pair, single)[0]
    # print(out1.shape, out2.shape, out3.shape)
    # assert torch.allclose(out1, out2, rtol=1e-4, atol=1e-5)
    # assert torch.allclose(out1, out3, rtol=1e-4, atol=1e-5)

    