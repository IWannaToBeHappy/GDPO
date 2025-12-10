#########################################################
#               BASELINES
##########################################################
# CUDA_VISIBLE_DEVICES=0 swift infer \
#     --model /workspace/model/hunyuan-1_8b-instruct \
#     --infer_backend vllm \
#     --max_new_tokens 2048 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_infer/hunyuan_base.jsonl \
#     --result_path /workspace/eval/experiments2/reverse_result/hunyuan_base.jsonl &

# CUDA_VISIBLE_DEVICES=1 swift infer \
#     --model /workspace/model/qwen3-1b \
#     --infer_backend vllm \
#     --model_type qwen3 \
#     --max_new_tokens 2048 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_infer/qwen3_base.jsonl \
#     --result_path /workspace/eval/experiments2/reverse_result/qwen3_base.jsonl &

# CUDA_VISIBLE_DEVICES=2 swift infer \
#     --model /workspace/model/llama-3_2-1B-instruct \
#     --infer_backend vllm \
#     --model_type llama \
#     --max_new_tokens 2048 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_infer/llama_base.jsonl \
#     --result_path /workspace/eval/experiments2/reverse_result/llama_base.jsonl &

# CUDA_VISIBLE_DEVICES=3 swift infer \
#     --model /workspace/model/glm-edge-1_5b-chat \
#     --infer_backend vllm \
#     --model_type glm_edge \
#     --max_new_tokens 2048 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_infer/glm_edge_base.jsonl \
#     --result_path /workspace/eval/experiments2/reverse_result/glm_edge_base.jsonl &

# wait

# CUDA_VISIBLE_DEVICES=0 swift infer \
#     --model /workspace/model/deepseek-coder-1_3b-instruct \
#     --infer_backend vllm \
#     --model_type deepseek \
#     --max_new_tokens 2048 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_infer/deepseek_base.jsonl \
#     --result_path /workspace/eval/experiments2/reverse_result/deepseek_base.jsonl &


# CUDA_VISIBLE_DEVICES=1 swift infer \
#     --model /workspace/model/gemma-2-2b-it \
#     --infer_backend vllm \
#     --max_new_tokens 2048 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_infer/gemma_base.jsonl \
#     --result_path /workspace/eval/experiments2/reverse_result/gemma_base.jsonl &

# wait
###############################################################
#                   RLHF
###############################################################

# CUDA_VISIBLE_DEVICES=0 swift infer \
#     --model /workspace/model/hunyuan-1_8b-instruct \
#     --infer_backend vllm \
#     --max_new_tokens 2048 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_infer/arithmetic_hunyuan_r1.jsonl \
#     --result_path /workspace/eval/experiments2/reverse_result/arithmetic_hunyuan_r1.jsonl \
#     --adapters /workspace/output/paper/arithmetic/hunyuan-1_8b-instruct-r1/v0-20251004-124521/checkpoint-1250 &

# CUDA_VISIBLE_DEVICES=1 swift infer \
#     --model /workspace/model/hunyuan-1_8b-instruct \
#     --infer_backend vllm \
#     --max_new_tokens 2048 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_infer/arithmetic_hunyuan_r2.jsonl \
#     --result_path /workspace/eval/experiments2/reverse_result/arithmetic_hunyuan_r2.jsonl \
#     --adapters /workspace/output/paper/arithmetic/hunyuan-1_8b-instruct-r2/v1-20251005-095157/checkpoint-1250 &

# CUDA_VISIBLE_DEVICES=2 swift infer \
#     --model /workspace/model/hunyuan-1_8b-instruct \
#     --infer_backend vllm \
#     --max_new_tokens 2048 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_infer/geometric_hunyuan_r1.jsonl \
#     --result_path /workspace/eval/experiments2/reverse_result/geometric_hunyuan_r1.jsonl \
#     --adapters /workspace/output/paper/geometric/hunyuan-1_8b-instruct-r1/v1-20251004-220157/checkpoint-1250 &

# CUDA_VISIBLE_DEVICES=3 swift infer \
#     --model /workspace/model/hunyuan-1_8b-instruct \
#     --infer_backend vllm \
#     --max_new_tokens 2048 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_infer/geometric_hunyuan_r2.jsonl \
#     --result_path /workspace/eval/experiments2/reverse_result/geometric_hunyuan_r2.jsonl \
#     --adapters /workspace/output/paper/geometric/hunyuan-1_8b-instruct-r2/v0-20251005-200556/checkpoint-1250 &

# wait

rm /workspace/eval/experiments2/reverse_result/geometric_deepseek_r2.jsonl
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model /workspace/model/deepseek-coder-1_3b-instruct \
    --model_type deepseek \
    --infer_backend vllm \
    --max_new_tokens 2048 \
    --temperature 0.0 \
    --val_dataset /workspace/eval/experiments2/reverse_infer/geometric_deepseek_r2.jsonl \
    --result_path /workspace/eval/experiments2/reverse_result/geometric_deepseek_r2.jsonl \
    --adapters /workspace/output/paper/geometric/deepseek-coder-1_3b-instruct-r2/v0-20251008-111044/checkpoint-1250 &

rm /workspace/eval/experiments2/reverse_result/geometric_deepseek_r1.jsonl
CUDA_VISIBLE_DEVICES=1 swift infer \
    --model /workspace/model/deepseek-coder-1_3b-instruct \
    --model_type deepseek \
    --infer_backend vllm \
    --max_new_tokens 2048 \
    --temperature 0.0 \
    --val_dataset /workspace/eval/experiments2/reverse_infer/geometric_deepseek_r1.jsonl \
    --result_path /workspace/eval/experiments2/reverse_result/geometric_deepseek_r1.jsonl \
    --adapters /workspace/output/paper/geometric/deepseek-coder-1_3b-instruct-r1/v0-20251008-221240/checkpoint-2500 &

# arithmetic deepseek r2
rm /workspace/eval/experiments2/reverse_result/arithmetic_deepseek_r2.jsonl
CUDA_VISIBLE_DEVICES=2 swift infer \
    --model /workspace/model/deepseek-coder-1_3b-instruct \
    --model_type deepseek \
    --infer_backend vllm \
    --max_new_tokens 2048 \
    --temperature 0.0 \
    --val_dataset /workspace/eval/experiments2/reverse_infer/arithmetic_deepseek_r2.jsonl \
    --result_path /workspace/eval/experiments2/reverse_result/arithmetic_deepseek_r2.jsonl \
    --adapters /workspace/output/paper/arithmetic/deepseek-coder-1_3b-instruct-r2/v0-20251009-163545/checkpoint-1250 &

# arithmetic deepseek r1
rm /workspace/eval/experiments2/reverse_result/arithmetic_deepseek_r1.jsonl
CUDA_VISIBLE_DEVICES=3 swift infer \
    --model /workspace/model/deepseek-coder-1_3b-instruct \
    --model_type deepseek \
    --infer_backend vllm \
    --max_new_tokens 2048 \
    --temperature 0.0 \
    --val_dataset /workspace/eval/experiments2/reverse_infer/arithmetic_deepseek_r1.jsonl \
    --result_path /workspace/eval/experiments2/reverse_result/arithmetic_deepseek_r1.jsonl \
    --adapters /workspace/output/paper/arithmetic/deepseek-coder-1_3b-instruct-r1/v1-20251009-102746/checkpoint-1250 &

# nokl deepseek
rm /workspace/eval/experiments2/reverse_result/nokl_deepseek_r2.jsonl
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model /workspace/model/deepseek-coder-1_3b-instruct \
    --model_type deepseek \
    --infer_backend vllm \
    --max_new_tokens 2048 \
    --temperature 0.0 \
    --val_dataset /workspace/eval/experiments2/reverse_infer/nokl_deepseek_r2.jsonl \
    --result_path /workspace/eval/experiments2/reverse_result/nokl_deepseek_r2.jsonl \
    --adapters /workspace/output/paper/no_think/deepseek-coder-1_3b-instruct-r2/v0-20251009-223307/checkpoint-1250 &

##################################################
#
#            CCSD 
##################################################

# CUDA_VISIBLE_DEVICES=0 swift infer \
#     --model /workspace/model/hunyuan-1_8b-instruct \
#     --infer_backend vllm \
#     --max_new_tokens 2048 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_infer/ccsd_hunyuan_base.jsonl \
#     --result_path /workspace/eval/experiments2/reverse_result/ccsd_hunyuan_base.jsonl &

# CUDA_VISIBLE_DEVICES=1 swift infer \
#     --model /workspace/model/qwen3-1b \
#     --infer_backend vllm \
#     --model_type qwen3 \
#     --max_new_tokens 2048 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_infer/ccsd_qwen3_base.jsonl \
#     --result_path /workspace/eval/experiments2/reverse_result/ccsd_qwen3_base.jsonl &

# CUDA_VISIBLE_DEVICES=2 swift infer \
#     --model /workspace/model/llama-3_2-1B-instruct \
#     --infer_backend vllm \
#     --model_type llama \
#     --max_new_tokens 2048 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_infer/ccsd_llama_base.jsonl \
#     --result_path /workspace/eval/experiments2/reverse_result/ccsd_llama_base.jsonl &

# CUDA_VISIBLE_DEVICES=3 swift infer \
#     --model /workspace/model/glm-edge-1_5b-chat \
#     --infer_backend vllm \
#     --model_type glm_edge \
#     --max_new_tokens 2048 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_infer/ccsd_glm_edge_base.jsonl \
#     --result_path /workspace/eval/experiments2/reverse_result/ccsd_glm_edge_base.jsonl &

# wait

# CUDA_VISIBLE_DEVICES=0 swift infer \
#     --model /workspace/model/deepseek-coder-1_3b-instruct \
#     --infer_backend vllm \
#     --model_type deepseek \
#     --max_new_tokens 2048 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_infer/ccsd_deepseek_base.jsonl \
#     --result_path /workspace/eval/experiments2/reverse_result/ccsd_deepseek_base.jsonl &


# CUDA_VISIBLE_DEVICES=1 swift infer \
#     --model /workspace/model/gemma-2-2b-it \
#     --infer_backend vllm \
#     --max_new_tokens 2048 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_infer/ccsd_gemma_base.jsonl \
#     --result_path /workspace/eval/experiments2/reverse_result/ccsd_gemma_base.jsonl &


# CUDA_VISIBLE_DEVICES=2 swift infer \
#     --model /workspace/model/hunyuan-1_8b-instruct \
#     --infer_backend vllm \
#     --max_new_tokens 2048 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_infer/ccsd_arithmetic_hunyuan_r1.jsonl \
#     --result_path /workspace/eval/experiments2/reverse_result/ccsd_arithmetic_hunyuan_r1.jsonl \
#     --adapters /workspace/output/paper/arithmetic/hunyuan-1_8b-instruct-r1/v0-20251004-124521/checkpoint-1250 &

# CUDA_VISIBLE_DEVICES=3 swift infer \
#     --model /workspace/model/hunyuan-1_8b-instruct \
#     --infer_backend vllm \
#     --max_new_tokens 2048 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_infer/ccsd_arithmetic_hunyuan_r2.jsonl \
#     --result_path /workspace/eval/experiments2/reverse_result/ccsd_arithmetic_hunyuan_r2.jsonl \
#     --adapters /workspace/output/paper/arithmetic/hunyuan-1_8b-instruct-r2/v1-20251005-095157/checkpoint-1250 &

# wait

# CUDA_VISIBLE_DEVICES=0 swift infer \
#     --model /workspace/model/hunyuan-1_8b-instruct \
#     --infer_backend vllm \
#     --max_new_tokens 2048 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_infer/ccsd_geometric_hunyuan_r1.jsonl \
#     --result_path /workspace/eval/experiments2/reverse_result/ccsd_geometric_hunyuan_r1.jsonl \
#     --adapters /workspace/output/paper/geometric/hunyuan-1_8b-instruct-r1/v1-20251004-220157/checkpoint-1250 &

# CUDA_VISIBLE_DEVICES=1 swift infer \
#     --model /workspace/model/hunyuan-1_8b-instruct \
#     --infer_backend vllm \
#     --max_new_tokens 2048 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_infer/ccsd_geometric_hunyuan_r2.jsonl \
#     --result_path /workspace/eval/experiments2/reverse_result/ccsd_geometric_hunyuan_r2.jsonl \
#     --adapters /workspace/output/paper/geometric/hunyuan-1_8b-instruct-r2/v0-20251005-200556/checkpoint-1250 &


