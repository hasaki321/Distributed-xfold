# 2025.5.25 NEW!
The fastest Alphafold3 implementation on CPU from SJTU has release their [repository](https://github.com/HPC-SJTU/xfold) and [kernels](https://github.com/HPC-SJTU/af3_kernels/tree/main), the fastest can be achieve 300s for 1200 token length on 5 Intel 8558p CPU.

# Distributed-xfold ( ASC25 final LZU )
A distributed implementation of xfold using dataparallel and tensorpallel, optimized for Intel CPU.

## Install guide

```bash
conda create -n daf3 python=3.12 -y
```
requirements: torch>=2.5.0, einops==0.8.1

For Intel cpus, check `install_guide/torch_with_mpi.md` for compiling a torch with mpi

For Nvidia GPUs, just pip install torch with cuda from the pip source
```bash
pip install torch==2.6.0 einops==0.8.1
```


```bash
git submoduel sync --recursive
git submoduel update --recursive
```
sync the required repositories and requirements

### AF3
```bash
cd alphafold3

pip install -e .
build data
```

### Distributed xfold
Install the distributed_xfold packages and the optimized cpp scripts.
```bash
python setup.py install
```

### Run the scripts

We have provided the example excution scripts for running the inferece pipeline

check `./inference_pipeline.sh` for more details.

The script split the whole inferece process into three stage: `preprocess`, `model inference`, `post process` 
