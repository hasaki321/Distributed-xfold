
export AF3_PREFIX=/home/hers22/HRS/Alphafold3

export INPUTS_PATH=$AF3_PREFIX/processed
export MODEL_INPUT_PATH=$AF3_PREFIX/model_inputs
export MODEL_OUTPUT_PATH=$AF3_PREFIX/model_outputs
export OUTPUTS_PATH=$AF3_PREFIX/final_outputs

export MODEL_PATH=$AF3_PREFIX/models
export JSON_NAME="37aa_2JO9.json"

export EVO_INTRA_THREAD=4

python run_inference.py \
    --intermediate_input_dir $MODEL_INPUT_PATH \
    --json_name $JSON_NAME \
    --raw_results_output_dir $MODEL_OUTPUT_PATH \
    --model_dir $MODEL_PATH \
    --num_diffusion_samples 4 \
    --device cpu \
    --xsmm 1 
    # --compile 1 \

python postprocess_model_output.py \
  --intermediate_input_dir $MODEL_INPUT_PATH \
  --raw_results_input_dir $MODEL_OUTPUT_PATH \
  --final_output_dir $OUTPUTS_PATH