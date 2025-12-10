#!/bin/bash
# 如果参数指定clean_first,则先清空并创建结果目录：infer_result reverse_infer reverse_result spmi_result
# 休眠4小时
# sleep 4h

if [ "$1" == "clean_first" ]; then
    rm -rf /workspace/eval/experiments3/reverse_infer
    rm -rf /workspace/eval/experiments3/reverse_result
    rm -rf /workspace/eval/experiments3/spmi_result
    mkdir -p /workspace/eval/experiments3/reverse_infer
    mkdir -p /workspace/eval/experiments3/reverse_result
    mkdir -p /workspace/eval/experiments3/spmi_result
fi

python /workspace/eval/experiments2/eval_result_to_infer.py
/workspace/eval/experiments2/reverse_code_gen.sh
/workspace/eval/experiments2/spmi.sh