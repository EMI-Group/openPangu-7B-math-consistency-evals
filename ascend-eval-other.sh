#!/bin/bash

set -ex
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_V1=1
# export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

############## SimpleRL ##############
# DATA_NAME="amc23,var_amc23,aime24,var_aime24,aime25,var_aime25"


MODEL_NAME_OR_PATH=${1:-"Qwen/Qwen2.5-MATH-7B"}
DATA_NAME=${2:-"amc23,var_amc23,aime24,var_aime24,aime25,var_aime25"}
N_SAMPLES=${3:-"1"}


declare -A MODEL_PROMPT_MAP=(
    ["Qwen/Qwen2.5-MATH-7B"]="qwen25-math-cot"
    ["PRIME-RL/Eurus-2-7B-PRIME"]="qwen25-math-cot"
    ["sail/Qwen2.5-Math-7B-Oat-Zero"]="qwen25-math-cot"
    ["hkust-nlp/Qwen-2.5-Math-7B-SimpleRL-Zoo"]="qwen25-math-cot"
    ["Skywork/Skywork-OR1-Math-7B"]="deepseek-r1"
    ["qihoo360/Light-R1-7B-DS"]="deepseek-r1"
)

PROMPT_TYPE="${MODEL_PROMPT_MAP[$MODEL_NAME_OR_PATH]}"

python3 -u math_eval.py \
        --model_name_or_path "${MODEL_NAME_OR_PATH}" \
        --data_name $DATA_NAME \
        --output_dir "score_pass_${N_SAMPLES}/${MODEL_NAME_OR_PATH}" \
        --prompt_type "${PROMPT_TYPE}" \
        --num_test_sample "-1" \
        --temperature 1.0 \
        --n_sampling ${N_SAMPLES} \
        --top_p 0.8 \
        --use_vllm \
        --save_outputs \
        --max_tokens_per_call 98304

