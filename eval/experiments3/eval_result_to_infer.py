# 将代码生成的eval_result转化为对偶任务的infer数据集,只转化output_dir中没有的文件
import jsonlines
import os
import re

input_root = "/workspace/eval/experiments3/infer_result"
output_root = "/workspace/eval/experiments3/reverse_infer"

input_files = []
output_files = []
for filename in os.listdir(input_root):
    if filename.endswith(".jsonl"):
        input_file = os.path.join(input_root, filename)
        output_file = os.path.join(output_root, filename)
        if not os.path.exists(output_file):
            input_files.append(input_file)
            output_files.append(output_file)

# input_file格式:{
#     "response": "",
#     "labels": null,
#     "logprobs": null,
#     "messages": [
#         {
#             "role": "user",
#             "content": "",
#             "loss": null
#         },
#         {
#             "role": "assistant",
#             "content": ""
#         }
#     ]
# }
# output_file格式:{"messages": [{"role": "system", "content": "..."},{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
system = "You are a helpful assistant that generates code. Given a code summary, provide the corresponding code implementation in {language}. warpper your code with ```."

for input_file,output_file in zip(input_files, output_files):
    with open(input_file, "r", encoding="utf-8") as infile, jsonlines.open(output_file, "w") as writer:
        for i,obj in enumerate(jsonlines.Reader(infile)):
            language = "java"
            response = obj["response"]
            # 去除可能的<think> </think>内容
            pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
            response = re.sub(pattern, '', response).strip()
            messages =  [ {"role": "system", "content": system.format(language=language)},
                    {"role": "user", "content":  response},
                    {"role": "assistant", "content": obj["messages"][0]["content"]}
                    ]
            writer.write({"messages": messages})