# 将java数据集转化为标准格式
import json

source_path = "/workspace/baseline/PromptCS-main/dataset/java/clean_train.jsonl"
target_path = "/workspace/eval/experiments3/dataset/train.jsonl"

with open(source_path, "r", encoding="utf-8") as src_file, open(target_path, "w", encoding="utf-8") as tgt_file:
    for line in src_file:
        data = json.loads(line)
        messages = {
            "messages":[
                {"role": "user", "content": data['clean_code']},
                {"role": "assistant", "content": data['clean_doc']}
                        ]

        }
        tgt_file.write(json.dumps(messages) + "\n")


source_path = "/workspace/baseline/PromptCS-main/dataset/java/clean_valid.jsonl"
target_path = "/workspace/eval/experiments3/dataset/valid.jsonl"

with open(source_path, "r", encoding="utf-8") as src_file, open(target_path, "w", encoding="utf-8") as tgt_file:
    for line in src_file:
        data = json.loads(line)
        messages = {
            "messages":[{"role": "user", "content": data['clean_code']}]
        }
        tgt_file.write(json.dumps(messages) + "\n")