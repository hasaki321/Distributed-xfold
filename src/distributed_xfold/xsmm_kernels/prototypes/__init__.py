import tpp_pytorch_extension._C

TRI_BLOCKSIZE = 32

QKV_BLOCKSIZE = 64
A_BLOCKSIZE = 64
Ak_BLOCKSIZE = 512
C_BLOCKSIZE = 64

from . import register_fake_kernels

from . import TriangleMultiplication
from . import GridSelfAttention
from . import Batched_DiffusionSelfAttention
from . import Batched_DiffusionCrossAttention
