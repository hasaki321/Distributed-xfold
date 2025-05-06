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
import random
import numpy as np
import torch
import torch.distributed as dist
import time
import os
import argparse
from typing import Type, Dict, Any, Optional, Callable, Union, Tuple, List

class ModelRunner:
    def __init__(self,
                 model_cls: Type[torch.nn.Module],
                 model_kwargs: Dict[str, Any],
                 dp_size: int,
                 tp_size: int,
                 weights_path: Union[str, pathlib.Path],
                 use_gpu_if_available: bool = True):
        """
        Initializes the ModelRunner, sets up device, distributed environment,
        and the model.
        """
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.weights_path = pathlib.Path(weights_path)
        self.use_gpu = use_gpu_if_available

        self.rank: int = 0
        self.world_size: int = 1
        self.device: str = 'cpu'
        self.backend: Optional[str] = None
        self.device_mesh: Optional[DeviceMesh] = None
        self.model: Optional[torch.nn.Module] = None
        self._is_compiled: bool = False
        self._uses_xsmm: bool = False

        self._setup_environment()
        self._initialize_model()

    def _setup_environment(self):
        """Determines device and initializes distributed environment if needed."""
        # 1. Device Detection
        if self.use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
            print(f"CUDA is available. Will attempt to use GPU(s).")
            if torch.cuda.device_count() < self.tp_size * self.dp_size and self.tp_size * self.dp_size > 1 : # Check if enough GPUs for world size
                 # This check is more relevant if each rank is intended for a unique GPU
                 print(f"Warning: World size ({self.tp_size * self.dp_size}) might exceed available CUDA devices ({torch.cuda.device_count()}). Ensure correct GPU assignment.")
        else:
            self.device = 'cpu'
            print(f"Using CPU.")

        # 2. Distributed Backend Selection and Initialization
        self.world_size = self.dp_size * self.tp_size
        if self.world_size > 1:
            if self.device == 'cuda':
                self.backend = 'nccl'
                print(f"Multi-GPU detected. Using NCCL backend.")
            else: # CPU
                # Check if MPI is available (basic check, might need more robust detection)
                if "OMPI_COMM_WORLD_RANK" in os.environ or "PMI_RANK" in os.environ or "MPI_ENABLED" in os.environ:
                    self.backend = 'mpi'
                    print(f"MPI environment detected. Using MPI backend for CPU.")
                else:
                    self.backend = 'gloo' # Fallback for CPU multi-processing on single node
                    print(f"No MPI detected. Using Gloo backend for CPU (suitable for single-node multi-process).")
            
            if not dist.is_initialized():
                print(f"Initializing distributed backend '{self.backend}' (world_size={self.world_size})...")
                # Assumes environment variables are set by launcher (mpirun, torchrun)
                dist.init_process_group(backend=self.backend, init_method='env://')
            
            self.rank = dist.get_rank()
            # Assign specific GPU if using CUDA and multiple GPUs per node
            if self.device == 'cuda':
                num_gpus_per_node = torch.cuda.device_count()
                local_rank = self.rank % num_gpus_per_node # Simple assignment
                torch.cuda.set_device(local_rank)
                self.device = f'cuda:{local_rank}' # Update device string
            print(f"Rank {self.rank}/{self.world_size} initialized on {self.device} with backend {self.backend}.")
        else:
            self.rank = 0
            print("Running in single-process mode (world_size=1).")

        self.device_mesh = DeviceMesh(self.world_size, self.dp_size, self.tp_size)

    def _initialize_model(self):
        """Instantiates, loads weights, syncs, and shards the model."""
        print(f"Rank {self.rank}: Instantiating model {self.model_cls.__name__} on {self.device}...")
        # Pass device_mesh if model expects it
        try:
            self.model = self.model_cls(device_mesh=self.device_mesh, **self.model_kwargs).to(self.device)
        except TypeError:
             print(f"Rank {self.rank}: Warning - {self.model_cls.__name__} might not accept 'device_mesh'. Instantiating without it.")
             self.model = self.model_cls(**self.model_kwargs).to(self.device)


        print(f"Rank {self.rank}: Loading weights from {self.weights_path}...")
        import_jax_weights_(self.model, self.weights_path) # Adapt as needed
        self.model.eval()

        if self.world_size > 1:
            self._sync_model_params()
            if self.tp_size > 1:
                self._shard_model_params()
            dist.barrier() # Ensure all setup is done

    def _sync_model_params(self):
        """Synchronizes model parameters from rank 0 to all other ranks."""
        if self.world_size <= 1: return
        print(f"Rank {self.rank}: Synchronizing model parameters...")
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)
        dist.barrier()
        print(f"Rank {self.rank}: Parameters synchronized.")

    def _shard_model_params(self):
        """Calls shard_params on modules that support it if TP is enabled."""
        if self.world_size <= 1 or self.tp_size <= 1: return
        print(f"Rank {self.rank}: Sharding model parameters for TP...")
        shard_count = 0
        for module in self.model.modules():
            if hasattr(module, 'use_tp') and not module.use_tp:
                 continue
            if hasattr(module, 'shard_params') and callable(module.shard_params):
                try:
                    module.shard_params()
                    shard_count += 1
                    # if self.rank == 0: print(f"  Sharded params for {type(module).__name__}")
                except Exception as e:
                    print(f"Rank {self.rank}: Error sharding {type(module).__name__}: {e}")
        print(f"Rank {self.rank}: Called shard_params on {shard_count} TP-aware modules.")

    def compile_model(self, enable_onednn_fusion: bool = True):
        """Compiles the model using torch.compile."""
        if self._is_compiled:
            print(f"Rank {self.rank}: Model already compiled.")
            return
        if self.device == 'cpu' and enable_onednn_fusion:
            torch.jit.enable_onednn_fusion(True)
            print(f"Rank {self.rank}: Enabled oneDNN fusion.")

        print(f"Rank {self.rank}: Compiling model...")
        self.model = torch.compile(self.model)
        self._is_compiled = True

        if self.world_size > 1: dist.barrier()

    def enable_xsmm(self):
        """Enables XSMM optimizations if modules support it."""
        if self._uses_xsmm:
            print(f"Rank {self.rank}: XSMM already enabled.")
            return
        print(f"Rank {self.rank}: Enabling XSMM if supported by modules...")
        xsmm_count = 0
        for module in self.model.modules():
            if hasattr(module, 'load_xsmm_params') and callable(module.load_xsmm_params):
                if hasattr(module, 'xsmm_forward') and callable(module.xsmm_forward):
                    try:
                        module.load_xsmm_params()
                        module.forward = module.xsmm_forward # Monkey-patch forward
                        xsmm_count += 1
                        # if self.rank == 0: print(f"  Enabled XSMM for {type(module).__name__}")
                    except Exception as e:
                         print(f"Rank {self.rank}: Error enabling XSMM for {type(module).__name__}: {e}")
                else:
                    if self.rank == 0: print(f"  Module {type(module).__name__} has load_xsmm_params but no xsmm_forward.")
        print(f"Rank {self.rank}: Enabled XSMM for {xsmm_count} modules.")
        self._uses_xsmm = True
        if self.world_size > 1: dist.barrier()

    @torch.no_grad()
    def run_inference(self, inputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Runs inference on the provided inputs and times it.
        Only rank 0 returns the output.
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        # Move inputs to the correct device (all ranks do this for their local model copy)
        # Assuming inputs are already on CPU from load_data, move to self.device
        inputs_on_device = pytree.tree_map_only(
            torch.Tensor,
            lambda x: x.to(self.device),
            inputs
        )

        print(f"Rank {self.rank}: Starting inference...")
        if self.world_size > 1: dist.barrier() # Sync before timing
        start_time = time.time()

        # Consider torch.amp.autocast if using mixed precision
        # with torch.autocast(device_type=self.device.split(':')[0], enabled=(self.device != 'cpu')):
        output = self.model(inputs_on_device)

        if self.world_size > 1: dist.barrier() # Sync after inference before measuring time on rank 0
        end_time = time.time()
        inference_time = end_time - start_time

        if self.rank == 0:
            print(f"--- Rank 0 Inference Report ---")
            print(f"Model: {self.model_cls.__name__}")
            print(f"Device: {self.device.upper()}")
            print(f"World Size: {self.world_size} (DP={self.dp_size}, TP={self.tp_size})")
            print(f"Compiled: {self._is_compiled}")
            print(f"XSMM enabled: {self._uses_xsmm}")
            print(f"Inference Time: {inference_time:.4f} sec")
            print(f"-----------------------------")
            return output
        return None # Other ranks return None

    def __del__(self):
        """Ensures distributed cleanup when the object is garbage collected."""
        if self.rank == 0 : print(f"ModelRunner for rank {self.rank} being destroyed.")
        # cleanup() # Be careful with cleanup in __del__ if processes might outlive the object

# --- Data Handling Functions (from your script) ---
def load_data(args_ns, sanitised_name: str) -> Dict[str, Any]:
    data_path = os.path.join(args_ns.intermediate_input_dir, f"{sanitised_name}/{sanitised_name}_seed_1_featurised.pt")
    featurised_example = torch.load(data_path, weights_only=False) # Set weights_only=True if applicable for security
    featurised_example = pytree.tree_map(
        lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x, # Handle numpy arrays
        utils.remove_invalidly_typed_feats(featurised_example)
    )
    # Keep on CPU for now, ModelRunner will move to device
    featurised_example = pytree.tree_map_only(
        torch.Tensor,
        lambda x: x.to(device='cpu'),
        featurised_example,
    )
    if 'deletion_mean' in featurised_example:
        featurised_example['deletion_mean'] = featurised_example['deletion_mean'].to(dtype=torch.float32)
    return featurised_example

def post_process(result: Dict[str, Any]) -> Dict[str, Any]:
    result = pytree.tree_map_only(
        torch.Tensor,
        lambda x: x.to(dtype=torch.float32) if x.dtype == torch.bfloat16 else x,
        result,
    )
    result = pytree.tree_map_only(
        torch.Tensor, lambda x: x.cpu().detach().numpy(), result)
    return result

def save_output(args_ns, sanitised_name: str, output_data: Dict[str, Any]):
    output_dir = os.path.join(args_ns.raw_results_output_dir, sanitised_name)
    os.makedirs(output_dir, exist_ok=True)

    # Assuming seed is 1 for now, adapt if needed
    seed_val = 1
    output_filename = f'{sanitised_name}_seed_{seed_val}_raw_results.pt'
    output_path = os.path.join(output_dir, output_filename)
    torch.save(output, os.path.join(output_dir, f'{sanitised_name}_seed_{1}_raw_results.pt'))
    print(f"Saving raw inference results for seed {1} to {output_path}")


# --- Main Application Logic ---
def main_app(args_ns: argparse.Namespace):
    # Set random seeds for reproducibility (all ranks if running distributed)
    seed = 1 # Or from args
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"Initial PyTorch Threads: Inter-op={torch.get_num_interop_threads()}, Intra-op={torch.get_num_threads()}")
    # Optionally set thread counts based on args or system properties here
    # torch.set_num_threads(desired_intra_threads)
    # torch.set_num_interop_threads(desired_inter_threads)

    # Model specific arguments for DistributeAlphaFold3
    model_constructor_args = {
        'num_recycles': args_ns.num_recycles,
        'num_samples': args_ns.num_diffusion_samples,
        'diffusion_steps': args_ns.diffusion_steps,
        # Add any other kwargs your DistributeAlphaFold3 needs from args_ns
        # Example: 'c_m': args_ns.c_m, 'c_z': args_ns.c_z, etc.
    }

    # Initialize ModelRunner
    runner = ModelRunner(
        model_cls=DistributeAlphaFold3, # Your distributed model class
        model_kwargs=model_constructor_args,
        dp_size=args_ns.dp,
        tp_size=args_ns.tp,
        weights_path=args_ns.model_dir, # Directory containing model weights
        use_gpu_if_available=(args_ns.device == 'cuda')
    )

    # Optional: Compile model
    if args_ns.compile:
        runner.compile_model()

    # Optional: Enable XSMM
    if args_ns.xsmm:
        runner.enable_xsmm()

    # Load input data (only rank 0 needs to do heavy lifting if broadcasted)
    # For simplicity, all ranks load metadata, runner handles broadcast
    sanitised_name = args_ns.json_name.split(".")[0].lower()
    # if runner.rank == 0 or not dist.is_initialized(): # Let rank 0 load, or single process
    print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}]: Loading data for {sanitised_name}...")
    featurised_example = load_data(args_ns, sanitised_name)

    # Run inference
    output_from_rank0 = runner.run_inference(featurised_example)

    # Post-process and save (only if rank 0 got output)
    if output_from_rank0 is not None: # This implies rank == 0
        processed_output = post_process(output_from_rank0)
        save_output(args_ns, sanitised_name, processed_output)

    # Explicit cleanup if ModelRunner doesn't handle it in __del__ or if script ends abruptly
    # cleanup() # ModelRunner.__del__ can call this, or main_app can.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed AlphaFold3 Inference Runner")
    parser.add_argument('-dp', default=1, type=int, help="Data Parallelism size")
    parser.add_argument('-tp', default=1, type=int, help="Tensor Parallelism size")
    parser.add_argument('--num_diffusion_samples', default=4, type=int, help="Number of diffusion samples") # Reduced for faster testing
    parser.add_argument('--num_recycles', default=10, type=int, help="Number of recycle iterations") # For model_kwargs
    parser.add_argument('--diffusion_steps', default=200, type=int, help="Number of diffusion steps") # For model_kwargs

    parser.add_argument('--json_name', required=True, type=str, help="Name of the input JSON file (without path)")
    parser.add_argument('--intermediate_input_dir', required=True, type=str, help="Directory for featurised inputs")
    parser.add_argument('--raw_results_output_dir', required=True, type=str, help="Directory for raw results output")
    parser.add_argument('--model_dir', required=True, type=str, help="Directory containing model weights")

    parser.add_argument('--compile', action='store_true', help="Enable torch.compile")
    parser.add_argument('--xsmm', action='store_true', help="Enable XSMM optimizations if available")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help="Device to use ('auto' detects GPU)")
    args = parser.parse_args()

    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # This script should be launched using mpirun or torchrun if world_size > 1
    # Example for mpirun (DP=2, TP=1, CPU):
    # mpirun -np 2 -x OMP_NUM_THREADS=26 python your_script.py -dp 2 -tp 1 --json_name ...
    # Example for torchrun (DP=2, TP=1, CPU, single node):
    # torchrun --nproc_per_node=2 your_script.py -dp 2 -tp 1 --json_name ...
    # Example for torchrun (DP=2, TP=2, 2 GPUs on 1 node):
    # torchrun --nproc_per_node=4 your_script.py -dp 2 -tp 2 --device cuda --json_name ...

    main_app(args)