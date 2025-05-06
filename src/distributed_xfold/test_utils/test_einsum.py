import torch

c = 32
n = 16

a = torch.randn((c, n, n))
b = torch.randn((c, n, n))

out = torch.einsum('ckj,cki->cij', a, b)        # [c_pair, N_token, N_token]

local_a1, local_a2 = torch.chunk(a, 2, dim=2)   # [c_pair, N_token, N_token / 2 ]

out1 = torch.einsum('ckj,cki->cij', local_a1, b)       # [c_pair, N_token, N_token / 2]
out2 = torch.einsum('ckj,cki->cij', local_a2, b)       # [c_pair, N_token, N_token / 2]
print(out1.shape, out2.shape)
assert torch.allclose(out, torch.cat((out1,out2), dim=2))

local_a1, local_a2 = torch.chunk(a, 2, dim=1)   # [c_pair, N_token / 2, N_token ]
out = torch.einsum('cik,cjk->cij', a, b)        # [c_pair, N_token, N_token]

out1 = torch.einsum('cik,cjk->cij', local_a1, b)       # [c_pair, N_token / 2, N_token]
out2 = torch.einsum('cik,cjk->cij', local_a2, b)       # [c_pair, N_token / 2, N_token]
print(out1.shape, out2.shape)
assert torch.allclose(out, torch.cat((out1,out2), dim=1))