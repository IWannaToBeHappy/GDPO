# promptcs
CUDA_VISIBLE_DEVICES=0 swift infer \
     --model /workspace/model/deepseek-coder-1_3b-instruct \
     --infer_backend vllm \
     --model_type deepseek \
     --max_new_tokens 2048 \
     --temperature 0.0 \
     --val_dataset /workspace/eval/experiments3/reverse_infer/promptcs.jsonl \
     --result_path /workspace/eval/experiments3/reverse_result/promptcs.jsonl &

# pt4code
CUDA_VISIBLE_DEVICES=1 swift infer \
     --model /workspace/model/deepseek-coder-1_3b-instruct \
     --infer_backend vllm \
     --model_type deepseek \
     --max_new_tokens 2048 \
     --temperature 0.0 \
     --val_dataset /workspace/eval/experiments3/reverse_infer/pt4code.jsonl \
     --result_path /workspace/eval/experiments3/reverse_result/pt4code.jsonl &

# gdpo
CUDA_VISIBLE_DEVICES=2 swift infer \
     --model /workspace/model/deepseek-coder-1_3b-instruct \
     --infer_backend vllm \
     --model_type deepseek \
     --max_new_tokens 2048 \
     --temperature 0.0 \
     --val_dataset /workspace/eval/experiments3/reverse_infer/gdpo.jsonl \
     --result_path /workspace/eval/experiments3/reverse_result/gdpo.jsonl &