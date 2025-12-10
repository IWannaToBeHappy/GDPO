# 将TL-CodeSum数据集转化为jsonl格式,只使用数据集test
import jsonlines
inputfile = "/workspace/dataset/TL-CodeSum/test/test.json"
outputfile = "/workspace/dataset/eval/tl_codesum_test.jsonl"
# {"api_seq": [...], "comment": "...", "code": "...", "id": ...} => {"query": "中国的首都是哪里？", "response": "中国的首都是北京"}
with open(inputfile, "r", encoding="utf-8") as infile, jsonlines.open(outputfile, "w") as writer:
    for obj in jsonlines.Reader(infile):
        messages =  { "query": obj["code"],
                "response": obj["comment"]
                }
        writer.write(messages)
