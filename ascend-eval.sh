#!/bin/bash

set -ex
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_V1=1
# export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

############## SimpleRL ##############
# DATA_NAME="amc23,var_amc23,aime24,var_aime24,aime25,var_aime25"
DATA_NAME=${1:-"amc23,var_amc23,aime24,var_aime24,aime25,var_aime25"}
N_SAMPLES=${2:-"16"}
THINK_MODE=${3:-"slow"}

python3 -u math_eval.py \
        --model_name_or_path /opt/pangu/openPangu-Embedded-7B-V1.1 \
        --data_name $DATA_NAME \
        --output_dir "score_pass_${N_SAMPLES}/openPangu-Embedded-7B-V1.1" \
        --prompt_type "pangu" \
        --num_test_sample "-1" \
        --temperature 1.0 \
        --n_sampling ${N_SAMPLES} \
        --top_p 0.8 \
        --use_vllm \
        --pangu_think_mode ${THINK_MODE} \
        --apply_chat_template \
        --save_outputs \
        --max_tokens_per_call 98304
