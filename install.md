# Install Guide

requirements:
- ucx
- openmpi with ucx support
- cmake=3.28
- torch repository
- intel-oneapi:
    - mkl
    - mkl-dnn
    - 



## Full process
---
### Miniconda
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3

source ~/miniconda3/bin/activate

conda init --all
```
---
### ENVs
```bash
export BASE_PATH=/home/hers22/HRS/Alphafold3/bins

export MPI_PATH=$BASE_PATH/ompi
export UCX_PATH=$BASE_PATH/ucx
export GCC_PATH=$BASE_PATH/gcc-11

mkdir $MPI_PATH && mkdir $UCX_PATH && mkdir $GCC_PATH
```
---
### GCC-11
```bash
wget https://ftp.gnu.org/gnu/gcc/gcc-11.3.0/gcc-11.3.0.tar.gz
tar -xzvf gcc-11.3.0.tar.gz
cd gcc-11.3.0

./contrib/download_prerequisites

mkdir build && cd build

../configure --prefix=$GCC_PATH --enable-languages=c,c++,fortran --disable-multilib

make -j$(nproc) && make install
```
---
### UCX
```bash
apt install libibverbs-dev ibverbs-utils rdma-core
apt install automake flex

wget https://github.com/openucx/ucx/releases/download/v1.18.0/ucx-1.18.0.tar.gz

tar -xzvf ucx-1.18.0.tar.gz
cd ucx-1.18.0
mkdir build && cd build
../contrib/configure-release --prefix=$UCX_PATH

# ../configure --prefix=$UCX_PATH --with-verbs=/usr/include/infiniband --with-rdmacm
make -j$(nproc) && make install

```
---
### MPI
```bash
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.8.tar.gz

tar -xzvf openmpi-4.1.8.tar.gz
 
cd openmpi-4.1.8
mkdir build && cd build
export CC=$GCC_PATH/bin/gcc 
export CXX=$GCC_PATH/bin/g++ 
export FC=$GCC_PATH/bin/gfortran 
export LD_LIBRARY_PATH=$GCC_PATH/lib64:$LD_LIBRARY_PATH
 
 
../configure --prefix=$MPI_PATH --with-ucx=$UCX_PATH
make -j 40 && make install
```
---
### oneapi
export ONEAPI_PATH=/home/hers22/intel/oneapi

#### mkl
```bash
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/dc93af13-2b3f-40c3-a41b-2bc05a707a80/intel-onemkl-2025.1.0.803_offline.sh
bash intel-onemkl-2025.1.0.803_offline.sh
```
#### mkldnn
```bash
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/9cf476b7-5b8b-4995-ac33-91a446bc0c6e/intel-onednn-2025.1.0.653_offline.sh
bash intel-onednn-2025.1.0.653_offline.sh
```
#### compiler
```bash
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/cd63be99-88b0-4981-bea1-2034fe17f5cf/intel-dpcpp-cpp-compiler-2025.1.0.573_offline.sh
bash intel-dpcpp-cpp-compiler-2025.1.0.573_offline.sh
```
<!-- #### oneccl
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/8f5d5e38-1626-41c1-9c20-44d966c43ae1/intel-oneccl-2021.15.0.401_offline.sh -->

---
### AF3
```bash
tar -xzvf alphafold3.tar.gz

cd alphafold3

pip install rdkit==2024.3.2 absl-py==2.1.0 zstandard==0.23.0 jax==0.4.34 chex==0.1.87 jaxtyping==0.2.34 typeguard==2.13.3 dm-haiku==0.0.13
CC=$GCC_PATH/bin/gcc CXX=$GCC_PATH/bin/g++ python setup.py install
# pip install -e .
build data
```

---
### Torch
```bash
conda create -p ./daf3 python=3.12 -y
conda activate ./daf3
conda install cmake ninja -y

git config --global url."https://githubfast.com/".insteadof "https://github.com/"
git clone https://github.com/pytorch/pytorch.git && cd pytorch
git checkout -f v2.5.0

git submodule sync --recursive
git submodule update --init --recursive

pip install -r requirements.txt


#设置编译选项
export CC=$GCC_PATH/bin/gcc 
export CXX=$GCC_PATH/bin/g++ 
export ONEAPI_PATH=/home/hers22/intel/oneapi

source $ONEAPI_PATH/setvars.sh --force
export PATH=$MPI_PATH/bin:$UCX_PATH/bin:$GCC_PATH/bin:$PATH
export LD_LIBRARY_PATH=$GCC_PATH/lib64:$UCX_PATH/lib:$MPI_PATH/lib:$LD_LIBRARY_PATH

export CMAKE_PREFIX_PATH=$ONEAPI_PATH/mkl/latest/:$CMAKE_PREFIX_PATH
export CMAKE_INCLUDE_PATH=$ONEAPI_PATH/mkl/latest/include:$CMAKE_INCLUDE_PATH
export LD_LIBRARY_PATH=$ONEAPI_PATH/mkl/latest/lib:$LD_LIBRARY_PATH

export USE_CUDA=OFF
export USE_CUDNN=OFF
export USE_CUSPARSELT=OFF
export USE_EXCEPTION_PTR=ON
export USE_GFLAGS=OFF
export USE_GLOG=OFF
export USE_DISTRIBUTED=ON
export USE_GLOO=ON
export USE_MKL=ON
export USE_MKLDNN=ON
export USE_MPI=ON
export USE_NCCL=OFF
export USE_NNPACK=ON
export USE_ROCM=OFF
export USR_SYCL=OFF
export USE_ROCM_KERNEL_ASSERT=OFF
export USE_XPU=OFF
export BLAS=MKL
export USE_OPENMP=ON

python setup.py clean
CMAKE_C_COMPILER=$(which mpicc) CMAKE_CXX_COMPILER=$(which mpicxx) python setup.py build develop

pip install einops==0.8.1
```
---
### Bashrc
```bash
# export BASE_PATH=/home/hers22/ASC25
# export ONEAPI_PATH=/home/hers22/intel/oneapi
# # export ONEAPI_PATH=/opt/intel
# source $ONEAPI_PATH/setvars.sh --force

# export MPI_PATH=$BASE_PATH/ompi
# export UCX_PATH=$BASE_PATH/ucx
# export GCC_PATH=$BASE_PATH/gcc-11

# export PATH=$MPI_PATH/bin:$UCX_PATH/bin:$GCC_PATH/bin:$PATH
# export LD_LIBRARY_PATH=$MPI_PATH/lib:$GCC_PATH/lib64:$UCX_PATH/lib:/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
# export LD_PRELOAD=$ONEAPI_PATH/compiler/latest/lib/libiomp5.so:$LD_PRELOAD

export BASE_PATH=/home/hers22/ASC25
export ONEAPI_PATH=/home/hers22/intel/oneapi
source $ONEAPI_PATH/setvars.sh --force

export MPI_PATH=$BASE_PATH/ompi
export UCX_PATH=$BASE_PATH/ucx

export PATH=$MPI_PATH/bin:$UCX_PATH/bin:$PATH
export LD_LIBRARY_PATH=$MPI_PATH/lib:$UCX_PATH/lib:/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
export LD_PRELOAD=$ONEAPI_PATH/compiler/latest/lib/libiomp5.so:$LD_PRELOAD
```

---
### Torch ccl
python -m pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install intel-extension-for-pytorch==2.5.0 oneccl_bind_pt==2.5.0 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/cn/

basekit_root=/opt/intel
source $basekit_root/ccl/latest/env/vars.sh


## performance tip
```bash
export OMP_SCHEDULE=STATIC
export OMP_PROC_BIND=CLOSE
export GOMP_CPU_AFFINITY="N-M"

export LD_PRELOAD=<path>/libiomp5.so:$LD_PRELOAD
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:$LD_PRELOAD
```

Similar to CPU affinity settings in GNU OpenMP, environment variables are provided in libiomp to control CPU affinity settings. KMP_AFFINITY binds OpenMP threads to physical processing units. KMP_BLOCKTIME sets the time, in milliseconds, that a thread should wait, after completing the execution of a parallel region, before sleeping. In most cases, setting KMP_BLOCKTIME to 1 or 0 yields good performances. The following commands show a common settings with Intel OpenMP Runtime Library.
```
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
```

```
OMP_MAX_ACTIVE_LEVELS=3
KMP_achtive_levels=3
```
