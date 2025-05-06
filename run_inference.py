from distributed_xfold.d_alphafold3 import DistributeAlphaFold3

import random
import numpy as np
import torch
import torch.distributed as dist
from distributed_xfold.distribute_utils import cleanup, DeviceMesh
import time
import os
import argparse
from typing import Type, Tuple, Dict, Any, Optional, Callable

import torch.utils._pytree as pytree
from alphafold3.model.components import utils
from xfold.params import import_jax_weights_
import pathlib

seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

print(f"Start Inference using {torch.get_num_interop_threads()}-inter-threads")
print(f"Start Inference using {torch.get_num_threads()}-intra-threads")

def distributed_enviorment_initialization(backend, init_method):
     # --- Distributed Initialization ---
    if not dist.is_initialized():
        if backend == 'mpi' or backend == 'nccl':
            # MPI/NCCL often rely on environment variables set by launchers (mpirun, srun, torchrun)
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

@torch.no_grad()
def run_distributed_inference(
    dist_model_cls: Type[torch.nn.Module],
    model_kwargs: Dict[str, Any],
    inputs,
    dp_size: int,
    tp_size: int,
    backend: str = 'mpi', # Or 'gloo', 'nccl'
    init_method: Optional[str] = None, # e.g., 'env://' or 'tcp://...'
    device: str = 'cpu', # Device to run on ('cpu' or 'cuda')
    _compile: bool = False,
):
    rank = -1
    world_size = -1
    d_output = None
    dist_time = 0.0

    distributed_enviorment_initialization(backend, init_method)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    expected_world_size = dp_size * tp_size
    if world_size != expected_world_size:
            raise RuntimeError(f"Launched {world_size} processes, but expected dp={dp_size} * tp={tp_size} = {expected_world_size} processes.")

    # --- Device Setup ---
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available.")
        # Basic device assignment, might need refinement for multi-node/multi-gpu
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        current_device = f'cuda:{local_rank}'
        print(f"Rank {rank}: Using device {current_device}")
    else:
        current_device = 'cpu'
        print(f"Rank {rank}: Using CPU")


    # --- Model Instantiation ---
    print(f"Rank {rank}: Instantiating models...")
    device_mesh = DeviceMesh(world_size=world_size, dp_size=dp_size, tp_size=tp_size)
    d_model = dist_model_cls(device_mesh=device_mesh, **model_kwargs).to(current_device)
    import_jax_weights_(d_model, pathlib.Path('/home/hers22/HRS/Alphafold3/models'))
    d_model.eval() # Set distributed model to eval mode

    # --- Shard Distributed Model Parameters ---
    if args.tp > 1:
        print(f"Rank {rank}: Checking for parameter sharding...")
        shard_count = 0
        for module in d_model.modules():
            if hasattr(module, 'use_tp') and use_tp and callable(module.shard_params):
                module.shard_params()
                shard_count += 1
                if rank == 0:
                    print(f"Sharded params for {type(module).__name__}")
        print(f"Rank {rank}: Found and called shard_params on {shard_count} modules.")
        dist.barrier()
    else:
        print(f"Rank {rank}: Not enabled sharding")

    if args.xsmm:
        print(f"Rank {rank}: Checking for xsmm parameter allocation...")
        xsmm_count = 0
        for module in d_model.modules():
            if hasattr(module, 'load_xsmm_params') and callable(module.load_xsmm_params):
                module.load_xsmm_params()
                module.forward = module.xsmm_forward
                xsmm_count += 1
                if rank == 0:
                    print(f"Loaded xsmm params for {type(module).__name__}")
        print(f"Rank {rank}: Found and called load_xsmm_params on {xsmm_count} modules.")
        dist.barrier()

    # --- Distributed Inference ---
    if _compile: 
        torch.jit.enable_onednn_fusion(True)
        d_model = torch.compile(d_model) 

    print(f"Rank {rank}: Starting distributed inference...")
    d_start = time.time()
    # with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
    d_output = d_model(inputs) # Assumes output is gathered on all ranks or handled internally
    dist.barrier() # Ensure all ranks finish inference
    d_end = time.time()
    dist_time = (d_end - d_start)
    print(f"Rank {rank}: Distributed inference finished in {dist_time:.2f} sec.")

    # --- Result Comparison (Rank 0 Only) ---
    if rank == 0:
        print(f"Distributed Inference Time: {dist_time:.4f} sec")

    # --- Cleanup ---
    cleanup()
    return d_output if rank == 0 else None

def load_data(args, sanitised_name):
    data_path = os.path.join(args.intermediate_input_dir, f"{sanitised_name}/{sanitised_name}_seed_1_featurised.pt")
    featurised_example = torch.load(data_path, weights_only=False)
    featurised_example = pytree.tree_map(
        torch.from_numpy, utils.remove_invalidly_typed_feats(
            featurised_example)
    )
    featurised_example = pytree.tree_map_only(
        torch.Tensor,
        lambda x: x.to(device='cpu'),
        featurised_example,
    )
    featurised_example['deletion_mean'] = featurised_example['deletion_mean'].to(
        dtype=torch.float32)
    return featurised_example

def post_process(result):
    result = pytree.tree_map_only(
        torch.Tensor,
        lambda x: x.to(
            dtype=torch.float32) if x.dtype == torch.bfloat16 else x,
        result,
    )
    result = pytree.tree_map_only(
        torch.Tensor, lambda x: x.cpu().detach().numpy(), result)
    return result

def save_output(args, sanitised_name, output):
    output_dir = os.path.join(args.raw_results_output_dir, sanitised_name)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'{sanitised_name}_seed_?_raw_results.pt')
    torch.save(output, os.path.join(output_dir, f'{sanitised_name}_seed_{1}_raw_results.pt'))
    print(f"Saving raw inference results for seed {1} to {output_path}")


def main(args):
    model_args = {
        'num_recycles': 10,
        'num_samples': args.num_diffusion_samples,
        'diffusion_steps': 200,
    }
    sanitised_name = args.json_name.split(".")[0].lower()
    featurised_example = load_data(args, sanitised_name)
    output = run_distributed_inference(
        dist_model_cls=DistributeAlphaFold3,
        model_kwargs=model_args,
        inputs=featurised_example,
        dp_size=args.dp,
        tp_size=args.tp,
        backend='mpi',
        device='cpu',
        _compile=args.compile,
    )

    if output is not None:
        output = post_process(output)
        save_output(args, sanitised_name, output)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', default=1, type=int)
    parser.add_argument('-tp', default=1, type=int)
    parser.add_argument('--num_diffusion_samples', default=5, type=int)
    parser.add_argument('--json_name', required=True, type=str)
    parser.add_argument('--intermediate_input_dir', required=True, type=str)
    parser.add_argument('--raw_results_output_dir', required=True, type=str)
    parser.add_argument('--model_dir', required=True, type=str)

    parser.add_argument('--compile', default=False, type=bool)
    parser.add_argument('--xsmm', default=False, type=bool)
    args = parser.parse_args()

    main(args)