# Llama-3.2-1B-instruct
# gemma-2-2b-it
# GLM-edge-1.5b-chat
# deepseek-coder-1.3b-instruct
# Phi-4-mini-instruct
# Hunyuan-1.8B-Instruct
# qwen3-1b

# /workspace/dataset/eval/ccsd_c_functions_all_data.jsonl
# /workspace/dataset/eval/tl_codesum_test.jsonl

CUDA_VISIBLE_DEVICES=2 swift eval \
    --model /workspace/model/qwen3-1b \
    --model_type qwen3 \
    --eval_backend Native \
    --infer_backend pt \
    --system "You are a helpful assistant that helps with code summarization. generate concise and accurate summaries for the provided code snippets.\n" \
    --eval_dataset general_qa \
    --eval_output_dir /workspace/eval/baseline/eval_output \
    --eval_limit 1000 \
    --max_batch_size 2 \
    --dataset_args '{"general_qa": {"local_path": "/workspace/dataset/eval", "subset_list": ["ccsd_c_functions_all_data","tl_codesum_test"]}}'

CUDA_VISIBLE_DEVICES=0 swift eval \
    --model /workspace/model/llama-3_2-1B-instruct \
    --model_type llama \
    --eval_backend Native \
    --infer_backend vllm \
    --system "You are a helpful assistant that helps with code summarization. generate concise and accurate summaries for the provided code snippets.\n" \
    --eval_dataset general_qa \
    --eval_output_dir /workspace/eval/baseline/eval_output \
    --eval_limit 1000 \
    --dataset_args '{"general_qa": {"local_path": "/workspace/dataset/eval", "subset_list": ["ccsd_c_functions_all_data","tl_codesum_test"]}}'

CUDA_VISIBLE_DEVICES=3 swift eval \
    --model /workspace/model/glm-edge-1_5b-chat \
    --model_type glm_edge \
    --eval_backend Native \
    --infer_backend pt \
    --system "You are a helpful assistant that helps with code summarization. generate concise and accurate summaries for the provided code snippets.\n" \
    --eval_dataset general_qa \
    --eval_output_dir /workspace/eval/baseline/eval_output \
    --eval_limit 1000 \
    --max_batch_size 2 \
    --dataset_args '{"general_qa": {"local_path": "/workspace/dataset/eval", "subset_list": ["ccsd_c_functions_all_data","tl_codesum_test"]}}'


CUDA_VISIBLE_DEVICES=3 swift eval \
    --model /workspace/model/deepseek-coder-1_3b-instruct \
    --model_type deepseek \
    --eval_backend Native \
    --infer_backend vllm \
    --system "You are a helpful assistant that helps with code summarization. generate concise and accurate summaries for the provided code snippets.\n" \
    --eval_dataset general_qa \
    --eval_output_dir /workspace/eval/baseline/eval_output \
    --eval_limit 1000 \
    --dataset_args '{"general_qa": {"local_path": "/workspace/dataset/eval", "subset_list": ["ccsd_c_functions_all_data","tl_codesum_test"]}}'

CUDA_VISIBLE_DEVICES=2 swift eval \
    --model /workspace/model/hunyuan-1_8b-instruct \
    --eval_backend Native \
    --infer_backend pt \
    --system "You are a helpful assistant that helps with code summarization. generate concise and accurate summaries for the provided code snippets.\n" \
    --eval_dataset general_qa \
    --eval_output_dir /workspace/eval/baseline/eval_output \
    --eval_limit 1000 \
    --max_batch_size 8 \
    --dataset_args '{"general_qa": {"local_path": "/workspace/dataset/eval", "subset_list": ["ccsd_c_functions_all_data","tl_codesum_test"]}}'



###########################################

swift eval \
    --model /workspace/model/phi-4-mini-instruct \
    --model_type phi4 \
    --eval_backend Native \
    --infer_backend vllm \
    --system "You are a helpful assistant that helps with code summarization. generate concise and accurate summaries for the provided code snippets.\n" \
    --eval_dataset general_qa \
    --eval_output_dir /workspace/eval/baseline/eval_output \
    --eval_limit 1000 \
    --dataset_args '{"general_qa": {"local_path": "/workspace/dataset/eval", "subset_list": ["ccsd_c_functions_all_data","tl_codesum_test"]}}'


#https://github.com/huggingface/transformers/issues/39427
TORCHDYNAMO_CACHE_SIZE_LIMIT=999 CUDA_VISIBLE_DEVICES=1 swift eval \
    --model /workspace/model/gemma-2-2b-it \
    --eval_backend Native \
    --infer_backend vllm \
    --vllm_gpu_memory_utilization 0.5 \
    --system "You are a helpful assistant that helps with code summarization. generate concise and accurate summaries for the provided code snippets.\n" \
    --eval_dataset general_qa \
    --eval_output_dir /workspace/eval/baseline/eval_output \
    --eval_limit 1000 \
    --dataset_args '{"general_qa": {"local_path": "/workspace/dataset/eval", "subset_list": ["ccsd_c_functions_all_data","tl_codesum_test"]}}'
