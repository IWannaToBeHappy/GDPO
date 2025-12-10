# 将代码生成的eval_result转化为对偶任务的infer数据集
import jsonlines

input_files = [
    "/workspace/eval/baseline/result/deepseek-coder-1_3b-instruct/eval_result/20250911-143234.jsonl",
    "/workspace/eval/baseline/result/gemma-2-2b-it/eval_result/20250911-215109.jsonl",
    "/workspace/eval/baseline/result/glm-edge-1_5b-chat/eval_result/20250911-115007.jsonl",
    "/workspace/eval/baseline/result/hunyuan-1_8b-instruct/eval_result/20250911-190535.jsonl",
    "/workspace/eval/baseline/result/llama-3_2-1B-instruct/eval_result/20250911-220357.jsonl",
    "/workspace/eval/baseline/result/phi-4-mini-instruct/eval_result/20250911-222129.jsonl",
    "/workspace/eval/baseline/result/qwen3-1b/eval_result/20250911-114848.jsonl"
]
output_files = [
    "/workspace/dataset/reverse_infer/deepseek-coder-1_3b-instruct.jsonl",
    "/workspace/dataset/reverse_infer/gemma-2-2b-it.jsonl",
    "/workspace/dataset/reverse_infer/glm-edge-1_5b-chat.jsonl",
    "/workspace/dataset/reverse_infer/hunyuan-1_8b-instruct.jsonl",
    "/workspace/dataset/reverse_infer/llama-3_2-1B-instruct.jsonl",
    "/workspace/dataset/reverse_infer/phi-4-mini-instruct.jsonl",
    "/workspace/dataset/reverse_infer/qwen3-1b.jsonl"
]

# input_file格式:{"response": "...", "infer_request": {"messages": [{"role": "user", "content": "..."}], "images": [], "audios": [], "videos": [], "tools": null, "objects": {}}}
# output_file格式:{"messages": [{"role": "system", "content": "..."},{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
system = "You are a helpful assistant that generates code. Given a code summary, provide the corresponding code implementation in {language}. warpper your code with ```."

for input_file,output_file in zip(input_files, output_files):
    with open(input_file, "r", encoding="utf-8") as infile, jsonlines.open(output_file, "w") as writer:
        for i,obj in enumerate(jsonlines.Reader(infile)):
            # 前1000行为C代码，后1000行为Java代码
            language = "C" if i < 1000 else "Java"
            messages =  [ {"role": "system", "content": system.format(language=language)},
                    {"role": "user", "content":  obj["response"]},
                    {"role": "assistant", "content": obj["infer_request"]["messages"][-1]["content"]}
                    ]
            writer.write({"messages": messages})