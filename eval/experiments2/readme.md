本实验验证主实验的效果

他们分别包括：
glm-edge-1_5b-chat
deepseek-coder-1_3b-instruct
qwen3-1b
llama-3_2-1B-instruct.jsonl
hunyuan-1_8b-instruct.jsonl
gemma-2-2b-it.jsonl

在codestar 10000样本上的spmi指标优化程度

训练设置差异：
1. 使用external_reverse_arithmetic或external_reverse_geometric
2. 添加FleschReadingEase与否

实验过程：
1. 运行train_script下的各训练脚本，训练出各模型于/workspace/output/paper下
2. 评测各训练脚本的spmi性能。
    2.1 运行/workspace/eval/experiments2/data/dataset.py 生成测试数据集
    2.2 运行infer.sh 进行模型对数据集的推理，生成摘要
    2.3 运行eval_result_to_infer.py 将摘要生成推理结果转化为代码生成数据集，结果保存在/workspace/eval/experiments2/reverse_infer中
    2.4 运行/workspace/eval/experiments2/reverse_code_gen.sh生成代码生成数据，推理结果保存在/workspace/eval/experiments2/reverse_result中
    2.5 运行spmi.sh计算推理结果的spmi得分，结果保存在spmi_result中
