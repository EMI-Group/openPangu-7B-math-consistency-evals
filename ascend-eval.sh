#!/bin/bash

set -ex
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_V1=1

############## SimpleRL ##############
# DATA_NAME="amc23,var_amc23,aime24,var_aime24,aime25,var_aime25"
DATA_NAME="amc23"

python3 -u math_eval.py \
        --model_name_or_path /opt/pangu/openPangu-Embedded-7B-V1.1 \
        --data_name $DATA_NAME \
        --output_dir "score_pass_16/openPangu-Embedded-7B-V1.1" \
        --prompt_type "pangu" \
        --num_test_sample "-1" \
        --temperature 0.6 \
        --n_sampling 16 \
        --top_p 1 \
        --use_vllm \
        --max_tokens_per_call 8192 \
        --apply_chat_template \
        --save_outputs
