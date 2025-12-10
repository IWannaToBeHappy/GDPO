rm /workspace/eval/experiments3/spmi_result/gdpo.jsonl
CUDA_VISIBLE_DEVICES=0 python /workspace/eval/spmi/spmi.py \
    --model /workspace/model/deepseek-coder-1_3b-instruct \
    --model_type deepseek \
    --infer_backend pt \
    --remove_unused_columns false \
    --max_batch_size 4 \
    --temperature 0.0 \
    --val_dataset /workspace/eval/experiments3/reverse_result/gdpo.jsonl \
    --result_path /workspace/eval/experiments3/spmi_result/gdpo.jsonl &

rm /workspace/eval/experiments3/spmi_result/promptcs.jsonl
CUDA_VISIBLE_DEVICES=1 python /workspace/eval/spmi/spmi.py \
    --model /workspace/model/deepseek-coder-1_3b-instruct \
    --model_type deepseek \
    --infer_backend pt \
    --remove_unused_columns false \
    --max_batch_size 4 \
    --temperature 0.0 \
    --val_dataset /workspace/eval/experiments3/reverse_result/promptcs.jsonl \
    --result_path /workspace/eval/experiments3/spmi_result/promptcs.jsonl &

rm /workspace/eval/experiments3/spmi_result/pt4code.jsonl
CUDA_VISIBLE_DEVICES=2 python /workspace/eval/spmi/spmi.py \
    --model /workspace/model/deepseek-coder-1_3b-instruct \
    --model_type deepseek \
    --infer_backend pt \
    --remove_unused_columns false \
    --max_batch_size 4 \
    --temperature 0.0 \
    --val_dataset /workspace/eval/experiments3/reverse_result/pt4code.jsonl \
    --result_path /workspace/eval/experiments3/spmi_result/pt4code.jsonl &

wait
python /workspace/eval/experiments3/show_spmi_result.py