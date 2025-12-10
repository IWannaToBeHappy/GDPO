"""
1. 读取/workspace/eval/baseline/eval_output/native/*/reviews/模型/数据集.jsonl中保存的各模型实验数据
示例：
    {
        "index": 0,
        "input": "**User**: \nExecMaterializesOutput(NodeTag plantype)\n{\n\tswitch (plantype)\n\t{\n\t\tcase T_Material:\n\t\tcase T_FunctionScan:\n\t\tcase T_TableFuncScan:\n\t\tcase T_CteScan:\n\t\tcase T_NamedTuplestoreScan:\n\t\tcase T_WorkTableScan:\n\t\tcase T_Sort:\n\t\t\treturn true;\n\n\t\tdefault:\n\t\t\tbreak;\n\t}\n\n\treturn false;\n}",
        "target": "does a plan type materialize its output",
        "sample_score": {
            "score": {
                "value": {
                    "bleu-1": 0.030769230769230767,
                    "bleu-2": 2.2250738585072626e-308,
                    "bleu-3": 2.2250738585072626e-308,
                    "bleu-4": 2.2250738585072626e-308,
                    "Rouge-1-R": 0.14285714285714285,
                    "Rouge-1-P": 0.029411764705882353,
                    "Rouge-1-F": 0.048780484973230384,
                    "Rouge-2-R": 0.0,
                    "Rouge-2-P": 0.0,
                    "Rouge-2-F": 0.0,
                    "Rouge-L-R": 0.14285714285714285,
                    "Rouge-L-P": 0.023255813953488372,
                    "Rouge-L-F": 0.039999997592000146
                },
                "extracted_prediction": "The function `ExecMaterializesOutput` checks if the given `plantype` (node type) requires materializing the output. It returns `true` for specific types like `T_Material`, `T_FunctionScan`, etc., and `false` for others. The function returns `false` by default if the type is not in the specified list.",
                "prediction": "The function `ExecMaterializesOutput` checks if the given `plantype` (node type) requires materializing the output. It returns `true` for specific types like `T_Material`, `T_FunctionScan`, etc., and `false` for others. The function returns `false` by default if the type is not in the specified list.",
                "explanation": null,
                "metadata": {},
                "main_score_name": "Rouge-L-R"
            },
            "sample_id": 0,
            "group_id": 0,
            "sample_metadata": {}
        }
    }
2. 根据bleu-1和Rouge-L-F值分别筛选出前10%和后10%的数据，并对每一部分采样20个样本
3. 将采样数据打乱顺序，问答对保存在/workspace/eval/experiments1/code_sum_pair/指标/模型/数据集.json中，标签保存在label.json中
"""
import os
import json
import random

root = "/workspace/eval/baseline/eval_output/native"

meta = {}
for _ in os.listdir(root):
    result_path = os.path.join(root, _,"reviews")
    for model in os.listdir(result_path):
        model_path = os.path.join(result_path, model)
        meta.setdefault(model, {})
        for dataset in os.listdir(model_path):
            dataset_path = os.path.join(model_path, dataset)
            meta[model][dataset] = dataset_path
    
for model,dataset in meta.items():
    for data, path in dataset.items():
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        samples = [json.loads(line) for line in lines]
        # 根据bleu-1和Rouge-L-F值分别筛选出前10%和后10%的数据，并对每一部分采样20个样本
        bleu_samples = samples.copy()
        rouge_samples = samples.copy()

        def _get_metric(sample, metric_name):
            try:
                return float(sample.get("sample_score", {}).get("score", {}).get("value", {}).get(metric_name, 0.0))
            except Exception:
                return 0.0
        bleu_samples = sorted(bleu_samples, key=lambda x: _get_metric(x, "bleu-1"))
        rouge_samples = sorted(rouge_samples, key=lambda x: _get_metric(x, "Rouge-L-F"))
        top_10_bleu = bleu_samples[int(len(bleu_samples)*0.9):]
        bottom_10_bleu = bleu_samples[:int(len(bleu_samples)*0.1)]
        top_10_rouge = rouge_samples[int(len(rouge_samples)*0.9):]
        bottom_10_rouge = rouge_samples[:int(len(rouge_samples)*0.1)]
        sampled_bleu = random.sample(top_10_bleu, min(50, len(top_10_bleu))) + random.sample(bottom_10_bleu, min(50, len(bottom_10_bleu)))
        sampled_rouge = random.sample(top_10_rouge, min(50, len(top_10_rouge))) + random.sample(bottom_10_rouge, min(50, len(bottom_10_rouge)))
        label_bleu = [s["index"] for s in sampled_bleu[:50]]
        label_rouge = [s["index"] for s in sampled_rouge[:50]]
        random.shuffle(sampled_bleu)
        random.shuffle(sampled_rouge)
        # 将采样数据打乱顺序，问答对保存在/workspace/eval/experiments1/code_sum_pair/指标/模型/数据集.json中，标签保存在label.json中
        save_root = f"/workspace/eval/experiments1/code_sum_pair"
        os.makedirs(save_root, exist_ok=True)
        for metric, sampled in [("bleu", sampled_bleu), ("rouge", sampled_rouge)]:
            save_path = os.path.join(save_root, metric, model)
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, f"{data}.json"), "w", encoding="utf-8") as f:
                # write as JSONL: one JSON object per line
                for s in sampled:
                    obj = {"index": s.get("index"), "input": s.get("input"), "extracted_prediction": s.get("sample_score", {}).get("score", {}).get('extracted_prediction')}
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            with open(os.path.join(save_path, f"{data}_label.json"), "w", encoding="utf-8") as f:
                if metric == "bleu":
                    json.dump(label_bleu, f, indent=4)
                else:
                    json.dump(label_rouge, f, indent=4)