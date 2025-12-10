###############################################
#           BASELINES
###############################################
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model /workspace/model/hunyuan-1_8b-instruct \
    --infer_backend vllm \
    --vllm_max_model_len 2048 \
    --system "You are a helpful assistant that generates code summaries. Given a piece of code, provide a concise summary." \
    --val_dataset /workspace/eval/experiments2/data/starcoderdata_c_val.jsonl \
    --result_path /workspace/eval/experiments2/infer_result/hunyuan_base.jsonl &

CUDA_VISIBLE_DEVICES=1 swift infer \
    --model /workspace/model/qwen3-1b \
    --model_type qwen3 \
    --infer_backend vllm \
    --vllm_max_model_len 2048 \
    --system "You are a helpful assistant that generates code summaries. Given a piece of code, provide a concise summary." \
    --val_dataset /workspace/eval/experiments2/data/starcoderdata_c_val.jsonl \
    --result_path /workspace/eval/experiments2/infer_result/qwen3_base.jsonl &

CUDA_VISIBLE_DEVICES=2 swift infer \
    --model /workspace/model/llama-3_2-1B-instruct \
    --model_type llama3_2 \
    --infer_backend vllm \
    --vllm_max_model_len 2048 \
    --system "You are a helpful assistant that generates code summaries. Given a piece of code, provide a concise summary." \
    --val_dataset /workspace/eval/experiments2/data/starcoderdata_c_val.jsonl \
    --result_path /workspace/eval/experiments2/infer_result/llama_base.jsonl &

CUDA_VISIBLE_DEVICES=3 swift infer \
    --model /workspace/model/glm-edge-1_5b-chat \
    --model_type glm_edge \
    --infer_backend vllm \
    --vllm_max_model_len 2048 \
    --system "You are a helpful assistant that generates code summaries. Given a piece of code, provide a concise summary." \
    --val_dataset /workspace/eval/experiments2/data/starcoderdata_c_val.jsonl \
    --result_path /workspace/eval/experiments2/infer_result/glm_edge_base.jsonl &

wait

CUDA_VISIBLE_DEVICES=0 swift infer \
    --model /workspace/model/deepseek-coder-1_3b-instruct \
    --model_type deepseek \
    --infer_backend vllm \
    --vllm_max_model_len 2048 \
    --system "You are a helpful assistant that generates code summaries. Given a piece of code, provide a concise summary." \
    --val_dataset /workspace/eval/experiments2/data/starcoderdata_c_val.jsonl \
    --result_path /workspace/eval/experiments2/infer_result/deepseek_base.jsonl &

CUDA_VISIBLE_DEVICES=1 swift infer \
    --model /workspace/model/gemma-2-2b-it \
    --infer_backend vllm \
    --vllm_max_model_len 2048 \
    --system "You are a helpful assistant that generates code summaries. Given a piece of code, provide a concise summary." \
    --val_dataset /workspace/eval/experiments2/data/starcoderdata_c_val.jsonl \
    --result_path /workspace/eval/experiments2/infer_result/gemma_base.jsonl &

wait

################################################
#               RLHF
################################################

# arithmetic hunyuan r1
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model /workspace/model/hunyuan-1_8b-instruct \
    --infer_backend vllm \
    --vllm_max_model_len 2048 \
    --system "You are a helpful assistant that generates code summaries. Given a piece of code, provide a concise summary." \
    --adapters /workspace/output/paper/arithmetic/hunyuan-1_8b-instruct-r1/v0-20251004-124521/checkpoint-1250 \
    --val_dataset /workspace/eval/experiments2/data/starcoderdata_c_val.jsonl \
    --result_path /workspace/eval/experiments2/infer_result/arithmetic_hunyuan_r1.jsonl &

# arithmetic hunyuan r2
CUDA_VISIBLE_DEVICES=1 swift infer \
    --model /workspace/model/hunyuan-1_8b-instruct \
    --infer_backend vllm \
    --vllm_max_model_len 2048 \
    --system "You are a helpful assistant that generates code summaries. Given a piece of code, provide a concise summary." \
    --adapters /workspace/output/paper/arithmetic/hunyuan-1_8b-instruct-r2/v1-20251005-095157/checkpoint-1250 \
    --val_dataset /workspace/eval/experiments2/data/starcoderdata_c_val.jsonl \
    --result_path /workspace/eval/experiments2/infer_result/arithmetic_hunyuan_r2.jsonl &

# geometric hunyuan r1
CUDA_VISIBLE_DEVICES=2 swift infer \
    --model /workspace/model/hunyuan-1_8b-instruct \
    --infer_backend vllm \
    --vllm_max_model_len 2048 \
    --system "You are a helpful assistant that generates code summaries. Given a piece of code, provide a concise summary." \
    --adapters /workspace/output/paper/geometric/hunyuan-1_8b-instruct-r1/v1-20251004-220157/checkpoint-1250 \
    --val_dataset /workspace/eval/experiments2/data/starcoderdata_c_val.jsonl \
    --result_path /workspace/eval/experiments2/infer_result/geometric_hunyuan_r1.jsonl &

# geometric hunyuan r2
CUDA_VISIBLE_DEVICES=3 swift infer \
    --model /workspace/model/hunyuan-1_8b-instruct \
    --infer_backend vllm \
    --vllm_max_model_len 2048 \
    --system "You are a helpful assistant that generates code summaries. Given a piece of code, provide a concise summary." \
    --adapters /workspace/output/paper/geometric/hunyuan-1_8b-instruct-r2/v0-20251005-200556/checkpoint-1250 \
    --val_dataset /workspace/eval/experiments2/data/starcoderdata_c_val.jsonl \
    --result_path /workspace/eval/experiments2/infer_result/geometric_hunyuan_r2.jsonl &

wait

# geometric deepseek r2
rm /workspace/eval/experiments2/infer_result/geometric_deepseek_r2.jsonl
rm /workspace/eval/experiments2/reverse_infer/geometric_deepseek_r2.jsonl
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model /workspace/model/deepseek-coder-1_3b-instruct \
    --model_type deepseek \
    --infer_backend vllm \
    --vllm_max_model_len 2048 \
    --temperature 0.0 \
    --system "You are a helpful assistant that generates code summaries. Given a piece of code, provide a concise summary." \
    --adapters /workspace/output/paper/geometric/deepseek-coder-1_3b-instruct-r2/v0-20251008-111044/checkpoint-1250 \
    --val_dataset /workspace/eval/experiments2/data/starcoderdata_c_val.jsonl \
    --result_path /workspace/eval/experiments2/infer_result/geometric_deepseek_r2.jsonl &

# geometric deepseek r1
rm /workspace/eval/experiments2/infer_result/geometric_deepseek_r1.jsonl
rm /workspace/eval/experiments2/reverse_infer/geometric_deepseek_r1.jsonl
CUDA_VISIBLE_DEVICES=1 swift infer \
    --model /workspace/model/deepseek-coder-1_3b-instruct \
    --model_type deepseek \
    --infer_backend vllm \
    --vllm_max_model_len 2048 \
    --temperature 0.0 \
    --system "You are a helpful assistant that generates code summaries. Given a piece of code, provide a concise summary." \
    --adapters /workspace/output/paper/geometric/deepseek-coder-1_3b-instruct-r1/v0-20251008-221240/checkpoint-2500 \
    --val_dataset /workspace/eval/experiments2/data/starcoderdata_c_val.jsonl \
    --result_path /workspace/eval/experiments2/infer_result/geometric_deepseek_r1.jsonl &

# arithmetic deepseek r2
rm /workspace/eval/experiments2/infer_result/arithmetic_deepseek_r2.jsonl
rm /workspace/eval/experiments2/reverse_infer/arithmetic_deepseek_r2.jsonl
CUDA_VISIBLE_DEVICES=2 swift infer \
    --model /workspace/model/deepseek-coder-1_3b-instruct \
    --model_type deepseek \
    --infer_backend vllm \
    --vllm_max_model_len 2048 \
    --temperature 0.0 \
    --system "You are a helpful assistant that generates code summaries. Given a piece of code, provide a concise summary." \
    --adapters /workspace/output/paper/arithmetic/deepseek-coder-1_3b-instruct-r2/v0-20251009-163545/checkpoint-1250 \
    --val_dataset /workspace/eval/experiments2/data/starcoderdata_c_val.jsonl \
    --result_path /workspace/eval/experiments2/infer_result/arithmetic_deepseek_r2.jsonl &

# arithmetic deepseek r1
rm /workspace/eval/experiments2/infer_result/arithmetic_deepseek_r1.jsonl
rm /workspace/eval/experiments2/reverse_infer/arithmetic_deepseek_r1.jsonl
CUDA_VISIBLE_DEVICES=3 swift infer \
    --model /workspace/model/deepseek-coder-1_3b-instruct \
    --model_type deepseek \
    --infer_backend vllm \
    --vllm_max_model_len 2048 \
    --temperature 0.0 \
    --system "You are a helpful assistant that generates code summaries. Given a piece of code, provide a concise summary." \
    --adapters /workspace/output/paper/arithmetic/deepseek-coder-1_3b-instruct-r1/v1-20251009-102746/checkpoint-1250 \
    --val_dataset /workspace/eval/experiments2/data/starcoderdata_c_val.jsonl \
    --result_path /workspace/eval/experiments2/infer_result/arithmetic_deepseek_r1.jsonl &

# nokl deepseek
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model /workspace/model/deepseek-coder-1_3b-instruct \
    --model_type deepseek \
    --infer_backend vllm \
    --vllm_max_model_len 2048 \
    --temperature 0.0 \
    --system "You are a helpful assistant that generates code summaries. Given a piece of code, provide a concise summary." \
    --adapters /workspace/output/paper/no_think/deepseek-coder-1_3b-instruct-r2/v0-20251009-223307/checkpoint-1250 \
    --val_dataset /workspace/eval/experiments2/data/starcoderdata_c_val.jsonl \
    --result_path /workspace/eval/experiments2/infer_result/nokl_deepseek_r2.jsonl &

#################################################
#         CCSD
#################################################

baseline
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model /workspace/model/hunyuan-1_8b-instruct \
    --infer_backend vllm \
    --vllm_max_model_len 2048 \
    --system "You are a helpful assistant that generates code summaries. Given a piece of code, provide a concise summary." \
    --val_dataset /workspace/dataset/eval/ccsd_c_functions_all_data.jsonl#5000 \
    --result_path /workspace/eval/experiments2/infer_result/ccsd_hunyuan_base.jsonl &

CUDA_VISIBLE_DEVICES=1 swift infer \
    --model /workspace/model/qwen3-1b \
    --model_type qwen3 \
    --infer_backend vllm \
    --vllm_max_model_len 2048 \
    --system "You are a helpful assistant that generates code summaries. Given a piece of code, provide a concise summary." \
    --val_dataset /workspace/dataset/eval/ccsd_c_functions_all_data.jsonl#5000 \
    --result_path /workspace/eval/experiments2/infer_result/ccsd_qwen3_base.jsonl &

CUDA_VISIBLE_DEVICES=2 swift infer \
    --model /workspace/model/llama-3_2-1B-instruct \
    --model_type llama \
    --infer_backend vllm \
    --vllm_max_model_len 2048 \
    --system "You are a helpful assistant that generates code summaries. Given a piece of code, provide a concise summary." \
    --val_dataset /workspace/dataset/eval/ccsd_c_functions_all_data.jsonl#5000 \
    --result_path /workspace/eval/experiments2/infer_result/ccsd_llama_base.jsonl &

CUDA_VISIBLE_DEVICES=3 swift infer \
    --model /workspace/model/glm-edge-1_5b-chat \
    --model_type glm_edge \
    --infer_backend vllm \
    --vllm_max_model_len 2048 \
    --system "You are a helpful assistant that generates code summaries. Given a piece of code, provide a concise summary." \
    --val_dataset /workspace/dataset/eval/ccsd_c_functions_all_data.jsonl#5000 \
    --result_path /workspace/eval/experiments2/infer_result/ccsd_glm_edge_base.jsonl &

wait

CUDA_VISIBLE_DEVICES=0 swift infer \
    --model /workspace/model/deepseek-coder-1_3b-instruct \
    --model_type deepseek \
    --infer_backend vllm \
    --vllm_max_model_len 2048 \
    --system "You are a helpful assistant that generates code summaries. Given a piece of code, provide a concise summary." \
    --val_dataset /workspace/dataset/eval/ccsd_c_functions_all_data.jsonl#5000 \
    --result_path /workspace/eval/experiments2/infer_result/ccsd_deepseek_base.jsonl &


CUDA_VISIBLE_DEVICES=1 swift infer \
    --model /workspace/model/gemma-2-2b-it \
    --infer_backend vllm \
    --vllm_max_model_len 2048 \
    --system "You are a helpful assistant that generates code summaries. Given a piece of code, provide a concise summary." \
    --val_dataset /workspace/dataset/eval/ccsd_c_functions_all_data.jsonl#5000 \
    --result_path /workspace/eval/experiments2/infer_result/ccsd_gemma_base.jsonl &

# arithmetic hunyuan r1
CUDA_VISIBLE_DEVICES=2 swift infer \
    --model /workspace/model/hunyuan-1_8b-instruct \
    --infer_backend vllm \
    --vllm_max_model_len 2048 \
    --system "You are a helpful assistant that generates code summaries. Given a piece of code, provide a concise summary." \
    --adapters /workspace/output/paper/arithmetic/hunyuan-1_8b-instruct-r1/v0-20251004-124521/checkpoint-1250 \
    --val_dataset /workspace/dataset/eval/ccsd_c_functions_all_data.jsonl#5000 \
    --result_path /workspace/eval/experiments2/infer_result/ccsd_arithmetic_hunyuan_r1.jsonl

# arithmetic hunyuan r2
CUDA_VISIBLE_DEVICES=3 swift infer \
    --model /workspace/model/hunyuan-1_8b-instruct \
    --infer_backend vllm \
    --vllm_max_model_len 2048 \
    --system "You are a helpful assistant that generates code summaries. Given a piece of code, provide a concise summary." \
    --adapters /workspace/output/paper/arithmetic/hunyuan-1_8b-instruct-r2/v1-20251005-095157/checkpoint-1250 \
    --val_dataset /workspace/dataset/eval/ccsd_c_functions_all_data.jsonl#5000 \
    --result_path /workspace/eval/experiments2/infer_result/ccsd_arithmetic_hunyuan_r2.jsonl

wait

# geometric hunyuan r1
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model /workspace/model/hunyuan-1_8b-instruct \
    --infer_backend vllm \
    --vllm_max_model_len 2048 \
    --system "You are a helpful assistant that generates code summaries. Given a piece of code, provide a concise summary." \
    --adapters /workspace/output/paper/geometric/hunyuan-1_8b-instruct-r1/v1-20251004-220157/checkpoint-1250 \
    --val_dataset /workspace/dataset/eval/ccsd_c_functions_all_data.jsonl#5000 \
    --result_path /workspace/eval/experiments2/infer_result/ccsd_geometric_hunyuan_r1.jsonl

# geometric hunyuan r2
CUDA_VISIBLE_DEVICES=1 swift infer \
    --model /workspace/model/hunyuan-1_8b-instruct \
    --infer_backend vllm \
    --vllm_max_model_len 2048 \
    --system "You are a helpful assistant that generates code summaries. Given a piece of code, provide a concise summary." \
    --adapters /workspace/output/paper/geometric/hunyuan-1_8b-instruct-r2/v0-20251005-200556/checkpoint-1250 \
    --val_dataset /workspace/dataset/eval/ccsd_c_functions_all_data.jsonl#5000 \
    --result_path /workspace/eval/experiments2/infer_result/ccsd_geometric_hunyuan_r2.jsonl