CUDA_VISIBLE_DEVICES=0 swift infer \
    --model /workspace/model/qwen3-1b \
    --model_type qwen3 \
    --infer_backend vllm \
    --max_new_tokens 2048 \
    --temperature 0.0 \
    --val_dataset /workspace/dataset/reverse_infer/qwen3-1b.jsonl \
    --result_path /workspace/eval/baseline/reverse_result/qwen3-1b.jsonl
    # --adapters /workspace/output/qwen3-1b-gspo-kl-reward/v4-20250908-020204/checkpoint-2000 \

CUDA_VISIBLE_DEVICES=1 swift infer \
    --model /workspace/model/llama-3_2-1B-instruct \
    --model_type llama \
    --infer_backend vllm \
    --max_new_tokens 2048 \
    --temperature 0.0 \
    --val_dataset /workspace/dataset/reverse_infer/llama-3_2-1B-instruct.jsonl \
    --result_path /workspace/eval/baseline/reverse_result/llama-3_2-1B-instruct.jsonl


CUDA_VISIBLE_DEVICES=2 swift infer \
    --model /workspace/model/glm-edge-1_5b-chat \
    --model_type glm_edge \
    --infer_backend vllm \
    --max_new_tokens 2048 \
    --temperature 0.0 \
    --val_dataset /workspace/dataset/reverse_infer/glm-edge-1_5b-chat.jsonl \
    --result_path /workspace/eval/baseline/reverse_result/glm-edge-1_5b-chat.jsonl


CUDA_VISIBLE_DEVICES=1 swift infer \
    --model /workspace/model/deepseek-coder-1_3b-instruct \
    --model_type deepseek \
    --infer_backend vllm \
    --max_new_tokens 2048 \
    --temperature 0.0 \
    --val_dataset /workspace/dataset/reverse_infer/deepseek-coder-1_3b-instruct.jsonl \
    --result_path /workspace/eval/baseline/reverse_result/deepseek-coder-1_3b-instruct.jsonl


CUDA_VISIBLE_DEVICES=0 swift infer \
    --model /workspace/model/hunyuan-1_8b-instruct \
    --infer_backend vllm \
    --max_new_tokens 2048 \
    --temperature 0.0 \
    --val_dataset /workspace/dataset/reverse_infer/hunyuan-1_8b-instruct.jsonl \
    --result_path /workspace/eval/baseline/reverse_result/hunyuan-1_8b-instruct.jsonl


CUDA_VISIBLE_DEVICES=2 swift infer \
    --model /workspace/model/phi-4-mini-instruct \
    --model_type phi4 \
    --infer_backend vllm \
    --max_new_tokens 2048 \
    --temperature 0.0 \
    --val_dataset /workspace/dataset/reverse_infer/phi-4-mini-instruct.jsonl \
    --result_path /workspace/eval/baseline/reverse_result/phi-4-mini-instruct.jsonl


CUDA_VISIBLE_DEVICES=0 swift infer \
    --model /workspace/model/gemma-2-2b-it \
    --infer_backend vllm \
    --max_new_tokens 2048 \
    --temperature 0.0 \
    --val_dataset /workspace/dataset/reverse_infer/gemma-2-2b-it.jsonl \
    --result_path /workspace/eval/baseline/reverse_result/gemma-2-2b-it.jsonl