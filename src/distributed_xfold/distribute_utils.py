import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
import os
import time
import socket
from contextlib import closing
from typing import List, Tuple, Optional, Any
import torch.nn as nn
import math

def find_free_port():
    """Finds a free port on localhost."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def setup_dist_env(
    backend: str = 'mpi', # Or 'gloo', 'nccl'
    init_method: Optional[str] = None, # e.g., 'env://' or 'tcp://...'
):
    if not dist.is_initialized():
        if backend == 'mpi' or backend == 'nccl':
                print(f"Initializing backend '{backend}' using init_method='env://'")
                dist.init_process_group(backend=backend, init_method='env://')
        elif backend == 'gloo':
            if init_method is None:
                print("Warning: Gloo backend selected without explicit init_method. Assuming 'env://'. Set RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT env vars.")
                dist.init_process_group(backend=backend, init_method='env://')
            else:
                print(f"Initializing backend '{backend}' using init_method='{init_method}'")
                dist.init_process_group(backend=backend, init_method=init_method)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

def setup_gloo(rank, world_size, port, host='127.0.0.1'):
    """Initializes the Gloo process group."""
    os.environ['MASTER_ADDR'] = host
    os.environ['MASTER_PORT'] = str(port)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    print(f"Rank {rank}: Gloo initialized.")

def cleanup():
    """Destroys the process group."""
    print(f"Rank {dist.get_rank()}: Cleaned up.")
    dist.destroy_process_group()


class DeviceMesh:
    def __init__(self, world_size, dp_size:int=1, tp_size:int=1):
        assert dp_size * tp_size == world_size
        flatten_mesh = torch.linspace(0, world_size-1, steps=world_size, dtype=torch.long)
        device_mesh = flatten_mesh.reshape((dp_size, tp_size))
        self.world_size = world_size
        self.device_mesh = device_mesh
        self.dp_size = dp_size
        self.tp_size = tp_size
        
    def get_rank_coords(self, rank):
        return (self.device_mesh == rank).nonzero()

    def get_dp_group(self, rank):
        rank_coord = self.get_rank_coords(rank)
        return dist.new_group(self.device_mesh[:, rank_coord[0, 1]].tolist())
    
    def get_dp_shard_num(self, rank):
        return self.get_rank_coords(rank)[0, 0]

    def get_tp_group(self, rank):
        rank_coord = self.get_rank_coords(rank)
        return dist.new_group((self.device_mesh[rank_coord[0, 0]]).tolist())

    def get_tp_shard_num(self, rank):
        return self.get_rank_coords(rank)[0, 1]

    def get_all_device(self):
        return dist.new_group(self.device_mesh.flatten().tolist())

def init_dist_info(self, device_mesh):
    self.device_mesh = device_mesh
    self.use_tp = device_mesh.tp_size > 1
    self.use_dp = device_mesh.dp_size > 1
    self.dp_size = self.device_mesh.dp_size
    self.tp_size = self.device_mesh.tp_size

    if dist.is_initialized():
        self.rank = dist.get_rank()
        self.dp_shard_group = self.device_mesh.get_dp_group(self.rank)
        self.tp_shard_group = self.device_mesh.get_tp_group(self.rank)
    else:
        self.rank = 0
        self.dp_shard_group = None
        self.tp_shard_group = None
    self.dp_shard_num = self.device_mesh.get_dp_shard_num(self.rank)
    self.tp_shard_num = self.device_mesh.get_tp_shard_num(self.rank)

class ShardInfo:
    def __init__(self, tensor_size, num_shard):
        self.tensor_size = tensor_size
        self.num_shard = num_shard
        self.shard_size_list, self.shard_slice_list, self.per_chunk_size, self.remain_chunk_size = self._shard_info()

    def _shard_info(self):
        assert self.num_shard <= self.tensor_size
        remains = self.tensor_size % self.num_shard != 0
        shard_size_list = None
        if remains:
            per_chunk_size = math.ceil(self.tensor_size / self.num_shard)
            remain_chunk_size = self.tensor_size % per_chunk_size
            shard_size_list = [per_chunk_size for _ in range(self.num_shard - 1)] + [remain_chunk_size]
        else:
            per_chunk_size = self.tensor_size // self.num_shard
            shard_size_list = [per_chunk_size for _ in range(self.num_shard)]
            remain_chunk_size = 0
        chunk_slice = torch.cumsum(torch.tensor([0] + shard_size_list, dtype=torch.long), 0).tolist()
        shard_slice_list = [slice(chunk_slice[i], chunk_slice[i+1]) for i in range(self.num_shard)]
        return shard_size_list, shard_slice_list, per_chunk_size, remain_chunk_size

    def get_shard_size_list(self): return self.shard_size_list
    def get_shard_slice_list(self): return self.shard_slice_list
    def get_per_chunk_size(self): return self.per_chunk_size
    def get_remain_chunk_size(self): return self.remain_chunk_size
    def get_num_shard(self): return self.num_shard
    def get_tensor_size(self): return self.tensor_size

def shard_linear(
    fnn: torch.nn.Linear,
    rank: int,
    device_mesh: DeviceMesh,
    out: bool = True,              # True: shard output dim (Column Parallel), False: shard input dim (Row Parallel)
) -> torch.nn.Linear:
    """
    Shards a torch.nn.Linear layer for Tensor Parallelism based on rank's position in the device mesh.
    Args:
        fnn: The original nn.Linear layer.
        rank: The global rank of the current process.
        device_mesh: A 2D tensor representing the process grid [dp_size, tp_size].
                     Ranks are typically assigned row-major.
        out: Wether shard output.
    Returns:
        A new nn.Linear layer containing the shard for the given rank.
    """
    dim = 0 if out else 1
    if not isinstance(fnn, torch.nn.Linear):
        raise TypeError("Input 'fnn' must be a torch.nn.Linear layer.")

    # Assuming rank appears only once, get its coordinates
    tp_rank = device_mesh.get_rank_coords(rank)[0, 1].item() # Column index (TP rank) - THIS is what we need

    # --- Get original layer properties ---
    tp_size = device_mesh.tp_size
    dtype = fnn.weight.dtype
    device = fnn.weight.device # Use the device of the original weights
    in_features = fnn.in_features
    out_features = fnn.out_features
    use_bias = fnn.bias is not None

    sharded_in_features = in_features
    sharded_out_features = out_features
    sharded_weight = None
    sharded_bias = None # Initialize as None

    # --- Perform Sharding ---
    if dim == 0: # Shard Output Dimension (Column Parallelism)
        if out_features % tp_size != 0:
            raise ValueError(f"Output dimension ({out_features}) not divisible by TP size ({tp_size}) for dim=0 sharding.")
        sharded_out_features = out_features // tp_size
        # Chunk weight along output dimension (dim 0)
        weight_chunks = fnn.weight.chunk(tp_size, dim=0)
        sharded_weight = weight_chunks[tp_rank]

        if use_bias:
            bias_chunks = fnn.bias.chunk(tp_size, dim=0)
            sharded_bias = bias_chunks[tp_rank]

    elif dim == 1: # Shard Input Dimension (Row Parallelism)
        if in_features % tp_size != 0:
            raise ValueError(f"Input dimension ({in_features}) not divisible by TP size ({tp_size}) for dim=1 sharding.")
        sharded_in_features = in_features // tp_size
        # Chunk weight along input dimension (dim 1)
        weight_chunks = fnn.weight.chunk(tp_size, dim=1)
        sharded_weight = weight_chunks[tp_rank]

        if use_bias:
            sharded_bias = fnn.bias # Each rank gets the full bias (or handles it post-AllReduce)

    # --- Create the new sharded linear layer ---
    new_layer_use_bias = use_bias and (sharded_bias is not None)

    new_fnn = torch.nn.Linear(
        in_features=sharded_in_features,
        out_features=sharded_out_features,
        bias=new_layer_use_bias, # Pass bool here
        dtype=dtype,
        device=device
    )
    # --- Assign sharded parameters ---
    new_fnn.weight = nn.Parameter(sharded_weight.clone()) # Clone to avoid modifying original tensor chunks
    if new_layer_use_bias:
        new_fnn.bias = nn.Parameter(sharded_bias.clone())

    return new_fnn
    
# @torch.compiler.disable
def all_to_all(
    input_tensor: torch.Tensor,  # The local tensor shard
    shard_dim: int,
    unshard_dim: int,
    group: Any | None = None,
    async_op: bool = False,
    custom_imp: bool = False
):
    world_size = dist.get_world_size(group=group) # Get size of the relevant group
    rank_in_group = dist.get_rank(group=group)     # Get rank within the group
    backend = torch.distributed.get_backend()

    input_chunks = list(torch.chunk(input_tensor, world_size, dim=shard_dim))
    input_chunks = [chunk.contiguous() for chunk in input_chunks]
    output_chunks = [torch.empty_like(input_chunks[0]) for _ in range(world_size)]

    async_worker = dist.all_to_all(output_chunks, input_chunks, group=group, async_op=async_op)
    if async_op: return async_worker
    else: return torch.cat(output_chunks, dim=unshard_dim)

@torch.compiler.disable
def all_gather_into_tensor(
    output_tensor: torch.Tensor, # The pre-allocated full tensor
    input_tensor: torch.Tensor,  # The local tensor shard
    group: Any | None = None,
    async_op: bool = False,
):
    world_size = dist.get_world_size(group=group) # Get size of the relevant group
    rank_in_group = dist.get_rank(group=group)     # Get rank within the group
    backend = torch.distributed.get_backend()

    input_tensor = input_tensor.contiguous()
    gather_list = [torch.empty_like(input_tensor) for _ in range(world_size)]

    # Perform the all_gather - this fills the gather_list
    work = dist.all_gather(gather_list, input_tensor, group=group, async_op=async_op) # Use contiguous input
    if async_op: return gather_list, work

    # Determine the shard dimension based on shapes (this is fragile, better to pass explicitly)
    shard_dim = torch.where(torch.tensor(output_tensor.shape) != torch.tensor(input_tensor.shape))[0]
    if len(shard_dim) > 1:
        raise ValueError("Could not determine shard dimension for manual concatenation in Gloo/MPI all_gather.")
    else:
        shard_dim = shard_dim.item()

    # Concatenate the gathered tensors into the pre-allocated output tensor
    full_cat_tensor = torch.cat(gather_list, dim=shard_dim)
    # Ensure shapes match before copying
    if output_tensor.shape == full_cat_tensor.shape:
        output_tensor.copy_(full_cat_tensor) # Copy data into the pre-allocated tensor
    else:
        raise RuntimeError(f"Shape mismatch after cat: {output_tensor.shape} vs {full_cat_tensor.shape}")

    return output_tensor # Return the filled output tensor
