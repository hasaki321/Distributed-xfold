# ================ Set Distribute Enviorment ===============

export BASE_PATH=/home/hers22/HRS/Alphafold3/bins
export MPI_PATH=$BASE_PATH/ompi
export UCX_PATH=$BASE_PATH/ucx
export GCC_PATH=$BASE_PATH/gcc-11
export ONEAPI_PATH=/home/hers22/intel/oneapi

source $ONEAPI_PATH/setvars.sh --force
export PATH=$MPI_PATH/bin:$UCX_PATH/bin:$GCC_PATH/bin:$PATH
export LD_LIBRARY_PATH=$MPI_PATH/lib:$GCC_PATH/lib64:$UCX_PATH/lib:$LD_LIBRARY_PATH

export LD_PRELOAD=$ONEAPI_PATH/compiler/latest/lib/libiomp5.so:$LD_PRELOAD
# export LD_PRELOAD=/home/hers22/HRS/distribute_practice/jemalloc/lib/libjemalloc.so:$LD_PRELOAD

export PYTHON_PATH=$(which python)


# ================ Config Model Env ===============
export AF3_PREFIX=/home/hers22/HRS/Alphafold3

export INPUTS_PATH=$AF3_PREFIX/processed
export MODEL_INPUT_PATH=$AF3_PREFIX/model_inputs
export MODEL_OUTPUT_PATH=$AF3_PREFIX/model_outputs
export OUTPUTS_PATH=$AF3_PREFIX/final_outputs

export MODEL_PATH=$AF3_PREFIX/models

export JSON_NAME="37aa_2JO9.json"
# export JSON_NAME="107aa_1TCE.json"
# export JSON_NAME="301aa_3DB6.json"
# export JSON_NAME="583aa_1CF3.json"
export JSON_NAME="740aa_4A5S.json"
# export JSON_NAME="1284aa_4XWK.json"
# export JSON_NAME="1491aa_5KIS.json"

# ================ Prepare Input ===============
# python prepare_model_input.py \
#   --json_path $INPUTS_PATH/$JSON_NAME \
#   --intermediate_output_dir $MODEL_INPUT_PATH \
#   --buckets=40,108,304,436,584,740,960,1024,1284,1292,1304,1492

# ================ Run Distribute Inference ===============
# export EVO_INTRA_THREAD=4 # 40
# export EVO_INTRA_THREAD=8 # 108
# export EVO_INTRA_THREAD=13 # 304
export EVO_INTRA_THREAD=24 # 512
# export EVO_INTRA_THREAD=26 # 512

      #  -x TORCH_LOGS="+dynamo" \
      #  -x TORCHDYNAMO_VERBOSE=1 \

# mpirun -np 2 \
#        -host localhost:1,n2:1 \
#        --mca pml ucx \
#        --mca btl ^openib \
#        --map-by node \
#        --report-bindings \
#        --bind-to board \
#        -x EVO_INTRA_THREAD=$EVO_INTRA_THREAD \
#        -x KMP_active_levels=3 \
#        -x OMP_MAX_ACTIVE_LEVELS=3 \
#        -x UCX_NET_DEVICES=all \
#        -x UCX_TLS=rc,ud,sm \
#        -x LD_LIBRARY_PATH \
#        -x LD_PRELOAD=$LD_PRELOAD \
#        -x PATH \
#        -x OMP_NUM_THREADS=52 \
#        -x MKL_NUM_THREADS=52 \
#        $PYTHON_PATH run_inference.py -dp 2 -tp 1 \
#           --intermediate_input_dir $MODEL_INPUT_PATH \
#           --json_name $JSON_NAME \
#           --raw_results_output_dir $MODEL_OUTPUT_PATH \
#           --model_dir $MODEL_PATH \
#           --num_diffusion_samples 4 \
#           --xsmm 1 
      #     --compile 1

mpirun -np 4 \
       -host localhost:2,n2:2 \
       --mca pml ucx \
       --mca btl ^openib \
       --map-by socket \
       --bind-to socket \
       -x KMP_STACKSIZE="4g" \
       -x EVO_INTRA_THREAD=$EVO_INTRA_THREAD \
       -x KMP_active_levels=3 \
       -x OMP_MAX_ACTIVE_LEVELS=3 \
       -x UCX_NET_DEVICES=all \
       -x UCX_TLS=rc,ud,sm \
       -x LD_LIBRARY_PATH \
       -x LD_PRELOAD=$LD_PRELOAD \
       -x PATH \
       -x OMP_NUM_THREADS=26 \
       -x MKL_NUM_THREADS=26 \
       $PYTHON_PATH run_inference.py -dp 4 -tp 1 \
          --intermediate_input_dir $MODEL_INPUT_PATH \
          --json_name $JSON_NAME \
          --raw_results_output_dir $MODEL_OUTPUT_PATH \
          --model_dir $MODEL_PATH \
          --num_diffusion_samples 4 \
          --compile 1 \
          --xsmm 1 

# ================ Postprocess Output ===============
json_base="${JSON_NAME%.json}"

python postprocess_model_output.py \
  --intermediate_input_dir $MODEL_INPUT_PATH \
  --raw_results_input_dir $MODEL_OUTPUT_PATH \
  --final_output_dir $OUTPUTS_PATH