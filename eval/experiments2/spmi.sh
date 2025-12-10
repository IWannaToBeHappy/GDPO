####################################################
#               BASELINES
####################################################

# CUDA_VISIBLE_DEVICES=0 python /workspace/eval/spmi/spmi.py \
#     --model /workspace/model/hunyuan-1_8b-instruct \
#     --infer_backend pt \
#     --remove_unused_columns false \
#     --max_batch_size 4 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_result/hunyuan_base.jsonl \
#     --result_path /workspace/eval/experiments2/spmi_result/hunyuan_base.jsonl &

# CUDA_VISIBLE_DEVICES=1 python /workspace/eval/spmi/spmi.py \
#     --model /workspace/model/qwen3-1b \
#     --model_type qwen3 \
#     --infer_backend pt \
#     --remove_unused_columns false \
#     --max_batch_size 4 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_result/qwen3_base.jsonl \
#     --result_path /workspace/eval/experiments2/spmi_result/qwen3_base.jsonl &

# CUDA_VISIBLE_DEVICES=2 python /workspace/eval/spmi/spmi.py \
#     --model /workspace/model/llama-3_2-1B-instruct \
#     --model_type llama3_2 \
#     --infer_backend pt \
#     --remove_unused_columns false \
#     --max_batch_size 4 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_result/llama_base.jsonl \
#     --result_path /workspace/eval/experiments2/spmi_result/llama_base.jsonl &

# CUDA_VISIBLE_DEVICES=3 python /workspace/eval/spmi/spmi.py \
#     --model /workspace/model/glm-edge-1_5b-chat \
#     --model_type glm_edge \
#     --infer_backend pt \
#     --remove_unused_columns false \
#     --max_batch_size 4 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_result/glm_edge_base.jsonl \
#     --result_path /workspace/eval/experiments2/spmi_result/glm_edge_base.jsonl &

# wait

# CUDA_VISIBLE_DEVICES=0 python /workspace/eval/spmi/spmi.py \
#     --model /workspace/model/deepseek-coder-1_3b-instruct \
#     --model_type deepseek \
#     --infer_backend pt \
#     --remove_unused_columns false \
#     --max_batch_size 4 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_result/deepseek_base.jsonl \
#     --result_path /workspace/eval/experiments2/spmi_result/deepseek_base.jsonl &


# CUDA_VISIBLE_DEVICES=1 python /workspace/eval/spmi/spmi.py \
#     --model /workspace/model/gemma-2-2b-it \
#     --infer_backend pt \
#     --remove_unused_columns false \
#     --max_batch_size 4 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_result/gemma_base.jsonl \
#     --result_path /workspace/eval/experiments2/spmi_result/gemma_base.jsonl &

# wait
# ################################################################
# #               RLHF
# #################################################################


# CUDA_VISIBLE_DEVICES=0 python /workspace/eval/spmi/spmi.py \
#     --model /workspace/model/hunyuan-1_8b-instruct \
#     --infer_backend pt \
#     --remove_unused_columns false \
#     --max_batch_size 4 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_result/arithmetic_hunyuan_r1.jsonl \
#     --result_path /workspace/eval/experiments2/spmi_result/arithmetic_hunyuan_r1.jsonl \
#     --adapters /workspace/output/paper/arithmetic/hunyuan-1_8b-instruct-r1/v0-20251004-124521/checkpoint-1250 &

# CUDA_VISIBLE_DEVICES=1 python /workspace/eval/spmi/spmi.py \
#     --model /workspace/model/hunyuan-1_8b-instruct \
#     --infer_backend pt \
#     --remove_unused_columns false \
#     --max_batch_size 4 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_result/arithmetic_hunyuan_r2.jsonl \
#     --result_path /workspace/eval/experiments2/spmi_result/arithmetic_hunyuan_r2.jsonl \
#     --adapters /workspace/output/paper/arithmetic/hunyuan-1_8b-instruct-r2/v1-20251005-095157/checkpoint-1250 &

# CUDA_VISIBLE_DEVICES=2 python /workspace/eval/spmi/spmi.py \
#     --model /workspace/model/hunyuan-1_8b-instruct \
#     --infer_backend pt \
#     --remove_unused_columns false \
#     --max_batch_size 4 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_result/geometric_hunyuan_r1.jsonl \
#     --result_path /workspace/eval/experiments2/spmi_result/geometric_hunyuan_r1.jsonl \
#     --adapters /workspace/output/paper/geometric/hunyuan-1_8b-instruct-r1/v1-20251004-220157/checkpoint-1250 &

# CUDA_VISIBLE_DEVICES=3 python /workspace/eval/spmi/spmi.py \
#     --model /workspace/model/hunyuan-1_8b-instruct \
#     --infer_backend pt \
#     --remove_unused_columns false \
#     --max_batch_size 4 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_result/geometric_hunyuan_r2.jsonl \
#     --result_path /workspace/eval/experiments2/spmi_result/geometric_hunyuan_r2.jsonl \
#     --adapters /workspace/output/paper/geometric/hunyuan-1_8b-instruct-r2/v0-20251005-200556/checkpoint-1250 &
# wait

rm /workspace/eval/experiments2/spmi_result/geometric_deepseek_r2.jsonl
CUDA_VISIBLE_DEVICES=0 python /workspace/eval/spmi/spmi.py \
    --model /workspace/model/deepseek-coder-1_3b-instruct \
    --model_type deepseek \
    --infer_backend pt \
    --remove_unused_columns false \
    --max_batch_size 4 \
    --temperature 0.0 \
    --val_dataset /workspace/eval/experiments2/reverse_result/geometric_deepseek_r2.jsonl \
    --result_path /workspace/eval/experiments2/spmi_result/geometric_deepseek_r2.jsonl \
    --adapters /workspace/output/paper/geometric/deepseek-coder-1_3b-instruct-r2/v0-20251008-111044/checkpoint-1250 &

rm /workspace/eval/experiments2/spmi_result/geometric_deepseek_r1.jsonl
CUDA_VISIBLE_DEVICES=1 python /workspace/eval/spmi/spmi.py \
    --model /workspace/model/deepseek-coder-1_3b-instruct \
    --model_type deepseek \
    --infer_backend pt \
    --remove_unused_columns false \
    --max_batch_size 4 \
    --temperature 0.0 \
    --val_dataset /workspace/eval/experiments2/reverse_result/geometric_deepseek_r1.jsonl \
    --result_path /workspace/eval/experiments2/spmi_result/geometric_deepseek_r1.jsonl \
    --adapters /workspace/output/paper/geometric/deepseek-coder-1_3b-instruct-r1/v0-20251008-221240/checkpoint-2500 &

rm /workspace/eval/experiments2/spmi_result/arithmetic_deepseek_r2.jsonl
CUDA_VISIBLE_DEVICES=2 python /workspace/eval/spmi/spmi.py \
    --model /workspace/model/deepseek-coder-1_3b-instruct \
    --model_type deepseek \
    --infer_backend pt \
    --remove_unused_columns false \
    --max_batch_size 4 \
    --temperature 0.0 \
    --val_dataset /workspace/eval/experiments2/reverse_result/arithmetic_deepseek_r2.jsonl \
    --result_path /workspace/eval/experiments2/spmi_result/arithmetic_deepseek_r2.jsonl \
    --adapters /workspace/output/paper/arithmetic/deepseek-coder-1_3b-instruct-r2/v0-20251009-163545/checkpoint-1250 &

rm /workspace/eval/experiments2/spmi_result/arithmetic_deepseek_r1.jsonl
CUDA_VISIBLE_DEVICES=3 python /workspace/eval/spmi/spmi.py \
    --model /workspace/model/deepseek-coder-1_3b-instruct \
    --model_type deepseek \
    --infer_backend pt \
    --remove_unused_columns false \
    --max_batch_size 4 \
    --temperature 0.0 \
    --val_dataset /workspace/eval/experiments2/reverse_result/arithmetic_deepseek_r1.jsonl \
    --result_path /workspace/eval/experiments2/spmi_result/arithmetic_deepseek_r1.jsonl \
    --adapters /workspace/output/paper/arithmetic/deepseek-coder-1_3b-instruct-r1/v1-20251009-102746/checkpoint-1250 &

rm /workspace/eval/experiments2/spmi_result/nokl_deepseek_r2.jsonl
CUDA_VISIBLE_DEVICES=0 python /workspace/eval/spmi/spmi.py \
    --model /workspace/model/deepseek-coder-1_3b-instruct \
    --model_type deepseek \
    --infer_backend pt \
    --remove_unused_columns false \
    --max_batch_size 4 \
    --temperature 0.0 \
    --val_dataset /workspace/eval/experiments2/reverse_result/nokl_deepseek_r2.jsonl \
    --result_path /workspace/eval/experiments2/spmi_result/nokl_deepseek_r2.jsonl \
    --adapters /workspace/output/paper/no_think/deepseek-coder-1_3b-instruct-r2/v0-20251009-223307/checkpoint-1250 &

######################################
#           CCSD
######################################


# CUDA_VISIBLE_DEVICES=2 python /workspace/eval/spmi/spmi.py \
#     --model /workspace/model/hunyuan-1_8b-instruct \
#     --infer_backend pt \
#     --remove_unused_columns false \
#     --max_batch_size 4 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_result/ccsd_hunyuan_base.jsonl \
#     --result_path /workspace/eval/experiments2/spmi_result/ccsd_hunyuan_base.jsonl &

# CUDA_VISIBLE_DEVICES=3 python /workspace/eval/spmi/spmi.py \
#     --model /workspace/model/qwen3-1b \
#     --model_type qwen3 \
#     --infer_backend pt \
#     --remove_unused_columns false \
#     --max_batch_size 4 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_result/ccsd_qwen3_base.jsonl \
#     --result_path /workspace/eval/experiments2/spmi_result/ccsd_qwen3_base.jsonl &

# wait

# CUDA_VISIBLE_DEVICES=0 python /workspace/eval/spmi/spmi.py \
#     --model /workspace/model/llama-3_2-1B-instruct \
#     --model_type llama3_2 \
#     --infer_backend pt \
#     --remove_unused_columns false \
#     --max_batch_size 4 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_result/ccsd_llama_base.jsonl \
#     --result_path /workspace/eval/experiments2/spmi_result/ccsd_llama_base.jsonl &

# CUDA_VISIBLE_DEVICES=1 python /workspace/eval/spmi/spmi.py \
#     --model /workspace/model/glm-edge-1_5b-chat \
#     --model_type glm_edge \
#     --infer_backend pt \
#     --remove_unused_columns false \
#     --max_batch_size 4 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_result/ccsd_glm_edge_base.jsonl \
#     --result_path /workspace/eval/experiments2/spmi_result/ccsd_glm_edge_base.jsonl &

# CUDA_VISIBLE_DEVICES=2 python /workspace/eval/spmi/spmi.py \
#     --model /workspace/model/deepseek-coder-1_3b-instruct \
#     --model_type deepseek \
#     --infer_backend pt \
#     --remove_unused_columns false \
#     --max_batch_size 4 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_result/ccsd_deepseek_base.jsonl \
#     --result_path /workspace/eval/experiments2/spmi_result/ccsd_deepseek_base.jsonl &

# CUDA_VISIBLE_DEVICES=3 python /workspace/eval/spmi/spmi.py \
#     --model /workspace/model/gemma-2-2b-it \
#     --infer_backend pt \
#     --remove_unused_columns false \
#     --max_batch_size 4 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_result/ccsd_gemma_base.jsonl \
#     --result_path /workspace/eval/experiments2/spmi_result/ccsd_gemma_base.jsonl &
# wait
# CUDA_VISIBLE_DEVICES=0 python /workspace/eval/spmi/spmi.py \
#     --model /workspace/model/hunyuan-1_8b-instruct \
#     --infer_backend pt \
#     --remove_unused_columns false \
#     --max_batch_size 4 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_result/ccsd_arithmetic_hunyuan_r1.jsonl \
#     --result_path /workspace/eval/experiments2/spmi_result/ccsd_arithmetic_hunyuan_r1.jsonl \
#     --adapters /workspace/output/paper/arithmetic/hunyuan-1_8b-instruct-r1/v0-20251004-124521/checkpoint-1250 &

# CUDA_VISIBLE_DEVICES=1 python /workspace/eval/spmi/spmi.py \
#     --model /workspace/model/hunyuan-1_8b-instruct \
#     --infer_backend pt \
#     --remove_unused_columns false \
#     --max_batch_size 4 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_result/ccsd_arithmetic_hunyuan_r2.jsonl \
#     --result_path /workspace/eval/experiments2/spmi_result/ccsd_arithmetic_hunyuan_r2.jsonl \
#     --adapters /workspace/output/paper/arithmetic/hunyuan-1_8b-instruct-r2/v1-20251005-095157/checkpoint-1250 &

# CUDA_VISIBLE_DEVICES=2 python /workspace/eval/spmi/spmi.py \
#     --model /workspace/model/hunyuan-1_8b-instruct \
#     --infer_backend pt \
#     --remove_unused_columns false \
#     --max_batch_size 4 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_result/ccsd_geometric_hunyuan_r1.jsonl \
#     --result_path /workspace/eval/experiments2/spmi_result/ccsd_geometric_hunyuan_r1.jsonl \
#     --adapters /workspace/output/paper/geometric/hunyuan-1_8b-instruct-r1/v1-20251004-220157/checkpoint-1250 &

# CUDA_VISIBLE_DEVICES=3 python /workspace/eval/spmi/spmi.py \
#     --model /workspace/model/hunyuan-1_8b-instruct \
#     --infer_backend pt \
#     --remove_unused_columns false \
#     --max_batch_size 4 \
#     --temperature 0.0 \
#     --val_dataset /workspace/eval/experiments2/reverse_result/ccsd_geometric_hunyuan_r2.jsonl \
#     --result_path /workspace/eval/experiments2/spmi_result/ccsd_geometric_hunyuan_r2.jsonl \
#     --adapters /workspace/output/paper/geometric/hunyuan-1_8b-instruct-r2/v0-20251005-200556/checkpoint-1250 &

# wait

python /workspace/eval/experiments2/show_spmi_result.py