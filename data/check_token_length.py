# 检查数据集的token长度，并进行百分位统计
import json
from transformers import AutoTokenizer
from tqdm import tqdm

dataset_path = "/workspace/dataset/train/starcoderdata_c_0.jsonl"
tokenizer_path = "/workspace/model/qwen3-1b"
#{"messages": [{"role": "user", "content": "..."}]}
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

with open(dataset_path, "r", encoding="utf-8") as f:
    lengths = []
    for line in tqdm(f):
        obj = json.loads(line)
        messages = obj["messages"]
        text = messages[-1]["content"]  # 只计算最后一条消息的长度
        tokens = tokenizer.encode(text, add_special_tokens=False)
        lengths.append(len(tokens))

lengths.sort()
total = len(lengths)
percentiles = [50, 75, 90, 95, 99, 99.9]
for p in percentiles:
    k = int(total * p / 100)
    print(f"{p}th percentile: {lengths[k]} tokens")


# 50th percentile: 186 tokens
# 75th percentile: 363 tokens
# 90th percentile: 722 tokens
# 95th percentile: 1135 tokens
# 99th percentile: 2977 tokens
# 99.9th percentile: 12365 tokens