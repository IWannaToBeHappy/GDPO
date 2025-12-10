# 将CCSD数据集转化为jsonl格式,只使用c_functions_all_data数据集
import jsonlines
inputfile = "/workspace/dataset/CCSD/c_functions_all_data.jsonl"
outputfile = "/workspace/dataset/eval/ccsd_c_functions_all_data.jsonl"
# {"file_path": "...", "function": "...", "summary": "..."} => {"query": "中国的首都是哪里？", "response": "中国的首都是北京"}
# 限制1w条
with open(inputfile, "r", encoding="utf-8") as infile, jsonlines.open(outputfile, "w") as writer:
    for obj in jsonlines.Reader(infile):
        messages = {"query": obj["function"],
           "response": obj["summary"]}
        