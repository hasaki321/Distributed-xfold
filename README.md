# Distributed-xfold
A distributed implementation of xfold using dataparallel and tensorpallel, optimized for Intel CPU

## Install guide

---
### AF3
```bash
cd alphafold3

pip install -e .
build data
```

---
### Bashrc
```bash
export BASE_PATH=/home/hers22/ASC25
export ONEAPI_PATH=/home/hers22/intel/oneapi
source $ONEAPI_PATH/setvars.sh --force

export MPI_PATH=$BASE_PATH/ompi
export UCX_PATH=$BASE_PATH/ucx

export PATH=$MPI_PATH/bin:$UCX_PATH/bin:$PATH
export LD_LIBRARY_PATH=$MPI_PATH/lib:$UCX_PATH/lib:/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
export LD_PRELOAD=$ONEAPI_PATH/compiler/latest/lib/libiomp5.so:$LD_PRELOAD
```
