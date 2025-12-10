CUDA_VISIBLE_DEVICES=0 python /workspace/eval/spmi/spmi.py \
    --model /workspace/model/qwen3-1b \
    --model_type qwen3 \
    --infer_backend pt \
    --remove_unused_columns false \
    --max_batch_size 4 \
    --temperature 0.0 \
    --val_dataset /workspace/eval/baseline/reverse_result/qwen3-1b.jsonl \
    --result_path /workspace/eval/experiments1/spmi_result/qwen3-1b.jsonl &
    # --adapters /workspace/output/qwen3-1b-gspo-kl-reward/v4-20250908-020204/checkpoint-2000 \

CUDA_VISIBLE_DEVICES=1 python /workspace/eval/spmi/spmi.py \
    --model /workspace/model/llama-3_2-1B-instruct \
    --model_type llama3_2 \
    --infer_backend pt \
    --remove_unused_columns false \
    --temperature 0.0 \
    --val_dataset /workspace/eval/baseline/reverse_result/llama-3_2-1B-instruct.jsonl \
    --result_path /workspace/eval/experiments1/spmi_result/llama-3_2-1B-instruct.jsonl &

CUDA_VISIBLE_DEVICES=2 python /workspace/eval/spmi/spmi.py \
    --model /workspace/model/glm-edge-1_5b-chat \
    --model_type glm_edge \
    --infer_backend pt \
    --remove_unused_columns false \
    --temperature 0.0 \
    --val_dataset /workspace/eval/baseline/reverse_result/glm-edge-1_5b-chat.jsonl \
    --result_path /workspace/eval/experiments1/spmi_result/glm-edge-1_5b-chat.jsonl &

CUDA_VISIBLE_DEVICES=3 python /workspace/eval/spmi/spmi.py \
    --model /workspace/model/deepseek-coder-1_3b-instruct \
    --model_type deepseek \
    --infer_backend pt \
    --remove_unused_columns false \
    --temperature 0.0 \
    --val_dataset /workspace/eval/baseline/reverse_result/deepseek-coder-1_3b-instruct.jsonl \
    --result_path /workspace/eval/experiments1/spmi_result/deepseek-coder-1_3b-instruct.jsonl &
wait

CUDA_VISIBLE_DEVICES=2,3 python /workspace/eval/spmi/spmi.py \
    --model /workspace/model/gemma-2-2b-it \
    --model_type gemma \
    --infer_backend pt \
    --remove_unused_columns false \
    --max_batch_size 1 \
    --temperature 0.0 \
    --val_dataset /workspace/eval/baseline/reverse_result/gemma-2-2b-it.jsonl \
    --result_path /workspace/eval/experiments1/spmi_result/gemma-2-2b-it.jsonl &

# transformers==4.56.2
CUDA_VISIBLE_DEVICES=0,1 python /workspace/eval/spmi/spmi.py \
    --model /workspace/model/hunyuan-1_8b-instruct \
    --model_type hunyuan \
    --infer_backend pt \
    --remove_unused_columns false \
    --temperature 0.0 \
    --val_dataset /workspace/eval/baseline/reverse_result/hunyuan-1_8b-instruct.jsonl \
    --result_path /workspace/eval/experiments1/spmi_result/hunyuan-1_8b-instruct.jsonl &

pip install transformers==4.53.3
CUDA_VISIBLE_DEVICES=0,1,2,3 python /workspace/eval/spmi/spmi.py \
    --model /workspace/model/phi-4-mini-instruct \
    --model_type phi4 \
    --infer_backend pt \
    --remove_unused_columns false \
    --max_batch_size 1 \
    --temperature 0.0 \
    --val_dataset /workspace/eval/baseline/reverse_result/phi-4-mini-instruct.jsonl \
    --result_path /workspace/eval/experiments1/spmi_result/phi-4-mini-instruct.jsonl &
pip install transformers --upgrade