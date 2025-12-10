# 本文件统计旧有数据集的摘要质量
import json

ccsd_file = "/workspace/dataset/eval/ccsd_c_functions_all_data.jsonl"
tl_codesum_file = "/workspace/dataset/eval/tl_codesum_test.jsonl"
starcoder_file = "/workspace/eval/experiments2/data/starcoderdata_c_val.jsonl"
# 指标一、统计摘要长度

for file in [ccsd_file, tl_codesum_file]:
    total_sum_len = 0
    total_code_len = 0
    count = 0
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            code = data['query']
            summary = data["response"]
            total_sum_len += len(summary.split())
            total_code_len += len(code.split())
            count += 1
    avg_sum_len = total_sum_len / count if count > 0 else 0
    avg_code_len = total_code_len / count if count > 0 else 0
    print(f"{file}: avg_summary_length={avg_sum_len},avg_code_length={avg_code_len} count={count}")
#/workspace/dataset/eval/ccsd_c_functions_all_data.jsonl: avg_summary_length=8.22082052035558, count=95281
# /workspace/dataset/eval/tl_codesum_test.jsonl: avg_summary_length=14.835781501032821, count=8714

total_code_len = 0
count = 0
with open(starcoder_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        code = data['messages'][0]['content']
        total_code_len += len(code.split())
        count += 1
avg_code_len = total_code_len / count if count > 0 else 0
print(f"{starcoder_file}: avg_code_length={avg_code_len} count={count}")
