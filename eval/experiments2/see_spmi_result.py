# 人工查看spmi结果与对应回答
import json
file = "/workspace/eval/experiments2/spmi_result/deepseek_base.jsonl"

with open(file, "r", encoding="utf-8") as f:
    # {"spmi": 0.5234375, "labels": "", "response": ""}
    # 先按spmi进行排序，然后从小到大展示spmi值和response内容
    data = [json.loads(line) for line in f]
    data = sorted(data, key=lambda x: x["spmi"])
    for item in data:
        print("="*20)
        print(f"{item['response']}")
        print(f"spmi: {item['spmi']},pmi: {item.get('pmi',None)}")
        a = input("Press Enter to see next...")
        if a.strip():
            print(f"labels: {item['labels']}")
        input("Press Enter to continue...")



"""
结论：
0分原因： 1.拒绝回答、索求更多信息、继续摘要任务
        2. 疑似训练数据复读(输出代码仅有函数名相同，内部逻辑完全不同)
50分原因：  1. 生成的代码本身信息量就少
            2. 模型陷入复读
            3. 疑似测试数据击中训练数据相似数据
100分原因： 1. 生成的代码与标签相似
            2. 模型陷入复读，但复读内容是代码关键信息
"""
