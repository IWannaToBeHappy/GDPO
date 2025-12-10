# confidence输出了每个样本的置信度值，现在计算模型整体置信度值
import json
import os

confidence_result_dir = "/workspace/eval/baseline/confidence_result"

for filename in os.listdir(confidence_result_dir):
    file_path = os.path.join(confidence_result_dir, filename)
    with open(file_path, "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f.readlines()]
        confidences = [res["confidence"] for res in results if "confidence" in res]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            print(f"{filename}: Average Confidence = {avg_confidence:.4f}")
        else:
            print(f"{filename}: No confidence values found.")

"""
glm-edge-1_5b-chat.jsonl: Average Confidence = 0.6669
deepseek-coder-1_3b-instruct.jsonl: Average Confidence = 0.2406
qwen3-1b.jsonl: Average Confidence = 0.7924
llama-3_2-1B-instruct.jsonl: Average Confidence = 0.2205
hunyuan-1_8b-instruct.jsonl: Average Confidence = 0.4913
phi-4-mini-instruct.jsonl: Average Confidence = 0.6314
gemma-2-2b-it.jsonl: Average Confidence = 0.7803
"""