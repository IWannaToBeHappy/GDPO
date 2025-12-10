# pt4code
rm /workspace/eval/experiments3/infer_result/pt4code.jsonl
rm /workspace/eval/experiments3/reverse_infer/pt4code.jsonl
swift infer \
    --model /workspace/baseline/PT4Code-main/summarization/model/v1-20251013-013357/checkpoint-4000 \
    --model_type deepseek \
    --infer_backend vllm \
    --vllm_max_model_len 2048 \
    --temperature 0.0 \
    --system "You are a helpful assistant that generates code summaries. Given a piece of code, provide a concise summary." \
    --val_dataset /workspace/eval/experiments3/dataset/valid.jsonl \
    --result_path /workspace/eval/experiments3/infer_result/pt4code.jsonl

# gdpo
rm /workspace/eval/experiments3/infer_result/gdpo.jsonl
rm /workspace/eval/experiments3/reverse_infer/gdpo.jsonl
CUDA_VISIBLE_DEVICES=2 swift infer \
    --model /workspace/model/deepseek-coder-1_3b-instruct \
    --model_type deepseek \
    --infer_backend vllm \
    --vllm_max_model_len 2048 \
    --vllm_gpu_memory_utilization 0.5 \
    --temperature 0.0 \
    --system "You are a helpful assistant that generates code summaries. Given a piece of code, provide a concise summary." \
    --adapters /workspace/output/paper/geometric/deepseek-coder-1_3b-instruct-r2/v1-20251008-191413/checkpoint-500 \
    --val_dataset /workspace/eval/experiments3/dataset/valid.jsonl \
    --result_path /workspace/eval/experiments3/infer_result/gdpo.jsonl