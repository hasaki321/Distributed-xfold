# export ONEAPI_PATH=/home/hers22/intel/oneapi
# # export ONEAPI_PATH=/opt/intel
# source $ONEAPI_PATH/setvars.sh --force

# export MPI_PATH=/home/hers22/HRS/distribute_practice/ompi
# export UCX_PATH=/home/hers22/HRS/distribute_practice/ucx
# export GCC_PATH=/home/hers22/HRS/distribute_practice/gcc-11

# export PATH=$MPI_PATH/bin:$UCX_PATH/bin:$GCC_PATH/bin:$PATH
# export LD_LIBRARY_PATH=$MPI_PATH/lib:$GCC_PATH/lib64:$UCX_PATH/lib:$LD_LIBRARY_PATH
# export LD_PRELOAD=/opt/intel/compiler/latest/lib/libiomp5.so:$LD_PRELOAD

# export LD_PRELOAD=/opt/intel/compiler/latest/lib/libiomp5.so:$LD_PRELOAD

export I_MPI_PORT_RANGE=40000:50000

basekit_root=/opt/intel
source $basekit_root/ccl/latest/env/vars.sh
source $basekit_root/mpi/latest/env/vars.sh

export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
# export MKL_DYNAMIC=FALSE
# export UCX_TLS=sm,rc_mlx5,dc_mlx5,ud_mlx5,self
export LD_PRELOAD=/opt/intel/mpi/latest/lib/libmpi_shm_heap_proxy.so
# export I_MPI_HYDRA_BOOTSTRAP=lsf
# export I_MPI_HYDRA_RMK=lsf
# export I_MPI_HYDRA_TOPOLIB=hwloc
export I_MPI_HYDRA_IFACE=ib0
# export I_MPI_PLATFORM=clx-ap
# export I_MPI_EXTRA_FILESYSTEM=1
# export I_MPI_EXTRA_FILESYSTEM_FORCE=gpfs
export I_MPI_FABRICS=shm:ofi
# export I_MPI_SHM=clx-ap
# export I_MPI_SHM_HEAP=1
export I_MPI_OFI_PROVIDER=mlx
# export I_MPI_PIN_CELL=core
export I_MPI_DEBUG=6

# …（省略其他）…

# 3. 启动命令
mpirun -n 2 -ppn 1 -hosts head:2 python test_ccl.py