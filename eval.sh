set -ex
export CUDA_VISIBLE_DEVICES=0,1

############## SimpleRL ##############
MODEL_NAME_OR_PATH_LIST=(Qwen/Qwen2.5-MATH-7B)
PROMPT_TYPE="qwen25-math-cot"

DATA_NAME="amc23,var_amc23,aime24,var_aime24,aime25,var_aime25"

for MODEL_NAME_OR_PATH in "${MODEL_NAME_OR_PATH_LIST[@]}"
do
    TOKENIZERS_PARALLELISM=false \
    python3 -u math_eval.py \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --data_name $DATA_NAME \
        --output_dir "score_pass_1/${MODEL_NAME_OR_PATH}" \
        --prompt_type $PROMPT_TYPE \
        --num_test_sample "-1" \
        --temperature 0.6 \
        --n_sampling 16 \
        --top_p 1 \
        --use_vllm \
        --max_tokens_per_call 8192 \
        # --apply_chat_template \
        --save_outputs
done