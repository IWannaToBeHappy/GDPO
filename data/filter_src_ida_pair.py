# 处理/workspace/dataset/源码IDA伪代码大模型回答码比较数据集.jsonl， 提取其中的IDA和源码对,去重后存入/workspace/dataset/source_ida_pair.csv
# 输入数据格式:{filehash:{"function_name":{"llm":{...},"ida":str,"source":str}}}
import json
import csv
from tqdm import tqdm
input_file = "/workspace/dataset/源码IDA伪代码大模型回答码比较数据集.json"
output_file = "/workspace/dataset/source_ida_pair.csv"

def load_data(input_file):
    data = []
    with open(input_file, "r", encoding="utf8") as f:
        all_data = json.load(f)
        for filehash, functions in tqdm(all_data.items(), desc="Processing files"):
            for func_name, contents in functions.items():
                if "ida" in contents and "source" in contents:
                    ida_code = contents["ida"].strip()
                    source_code = contents["source"].strip()
                    if ida_code and source_code:
                        data.append((source_code, ida_code, func_name))
    return data

def save_to_csv(data, output_file):
    unique_data = set(data)  # 去重
    with open(output_file, "w", encoding="utf8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["source_code", "ida_code", "function_name"])
        for row in unique_data:
            writer.writerow(row)

if __name__ == "__main__":
    data = load_data(input_file)
    save_to_csv(data, output_file)
    print(f"Saved {len(data)} unique source-ida pairs to {output_file}")