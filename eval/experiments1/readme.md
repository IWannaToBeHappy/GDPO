本文件夹对应实验一

# 1.各基线模型在BLEU、ROUGE-L指标上的表现
运行/workspace/eval/experiments/infer.sh脚本完成评估。
中间结果生成/workspace/eval/baseline/eval_output/native文件夹下数据。
评测结果生成在/workspace/eval/baseline/result

# 2.BLUE、ROUGE-L指标与人类一致性实验
运行/workspace/eval/experiments1/n-gram_humaneval.py生成待人类排序样本-摘要对。结果保存在/workspace/eval/experiments1/code_sum_pair下。
运行is_good_summary.py，与spmi实验一起运行

# 3.各基线模型在spmi指标上的表现
运行/workspace/eval/experiments1/eval_result_to_infer.py脚本 将摘要生成推理结果转化为代码生成数据集，结果保存在 /workspace/dataset/reverse_infer 中。
运行/workspace/eval/experiments1/reverse_code_gen.sh生成代码生成数据，推理结果保存在/workspace/eval/baseline/reverse_result中
运行/workspace/eval/experiments1/spmi.sh计算推理结果的spmi得分。评测结果保存在/workspace/eval/experiments1/spmi_result中，效果暂存于spmi_humaneval.py的注释中

# spmi指标与人类一致性实验
运行/workspace/eval/experiments1/spmi_humaneval.py生成人类排序样本-摘要对。结果保存在/workspace/eval/experiments1/code_sum_pair/spmi下
运行is_good_summary.py