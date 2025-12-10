# 从/workspace/dataset/train/starcoderdata_c_0.jsonl中选取1k条长度不超过1024的代码样本作为测评数据集
# 结果保存在/workspace/eval/experiments2/data/starcoderdata_c_val.jsonl中
import json
from transformers import AutoTokenizer
import random
from tqdm import tqdm

dataset_path = "/workspace/dataset/train/starcoderdata_c_0.jsonl"
output_path = "/workspace/eval/experiments2/data/starcoderdata_c_val.jsonl"
tokenizer_path = "/workspace/model/qwen3-1b"
#{"messages": [{"role": "user", "content": "..."}]}
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

random.seed(1998)

# 只选取的样本数
TARGET = 1000

def cal_token_len(content):
    tokens = tokenizer.encode(content, add_special_tokens=False)
    return len(tokens)

with open(dataset_path, "r", encoding="utf-8") as f, open(output_path, "w", encoding="utf-8") as out_f:
    lines = f.readlines()
    random.shuffle(lines)
    selected = []
    for line in tqdm(lines):
        data = json.loads(line)
        if "messages" not in data:
            continue
        if not isinstance(data["messages"], list) or len(data["messages"]) == 0:
            continue
        if data["messages"][-1]["role"] != "user":
            continue
        content = data["messages"][-1]["content"]
        if cal_token_len(content) <= 1024:
            selected.append(data)
        if len(selected) >= TARGET:
            break
    print(f"selected {len(selected)} samples")
    if len(selected) < TARGET:
        print(f"Warning: only found {len(selected)} samples < TARGET={TARGET}")
    # 再次确保最多写入 TARGET 条
    for i, item in enumerate(selected):
        if i >= TARGET:
            break
        out_f.write(json.dumps(item, ensure_ascii=False) + "\n")