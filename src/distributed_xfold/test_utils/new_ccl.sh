#!/bin/bash
set -e

basekit_root=/opt/intel
source $basekit_root/ccl/latest/env/vars.sh
source $basekit_root/mpi/latest/env/vars.sh

# 1. 指定一个足够宽的端口区间，确保 hydra_bstrap_proxy 能拿到真实端口
export I_MPI_PORT_RANGE=20000:20001
export MASTER_ADDR="198.18.18.254"
export MASTER_PORT="20001"
export FI_PROVIDER=tcp
# export I_MPI_HYDRA_IFACE=tcp

# 2. 强制 Hydra 用 SSH 打桩，避免 PMI 或环境继承问题
BOOTSTRAP="ssh"
HOSTS="head:2,n205:2"
NP=4
PPN=2

echo "=== Running mpiexec.hydra ==="
which mpirun
mpirun \
  -n $NP \
  -ppn $PPN \
  -f ./hostfile \
  -verbose \
  python test_ccl.py