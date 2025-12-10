"""
1. 读取/workspace/eval/experiments1/spmi_result/模型.jsonl中保存的各模型实验数据
示例：
    {
    "spmi": 0.8681235909461975,
    "labels": "void qede_recycle_rx_bd_ring(struct qede_rx_queue *rxq, u8 count)\n{\n\tstruct sw_rx_data *curr_cons;\n\n\tfor (; count > 0; count--) {\n\t\tcurr_cons = &rxq->sw_rx_ring[rxq->sw_rx_cons & NUM_RX_BDS_MAX];\n\t\tqede_reuse_page(rxq, curr_cons);\n\t\tqede_rx_bd_ring_consume(rxq);\n\t}\n}",
    "response": " \n\nHere is a simple implementation of the `qede_recycle_rx_bd_ring` function in C:\n\n```c\n#include \"qede_rx_bd_ring.h\"\n\nvoid qede_recycle_rx_bd_ring(qede_rx_queue *queue, int count) {\n    for (int i = 0; i < count; i++) {\n        qede_reuse_page(queue);\n        qede_rx_bd_ring_consume(queue);\n    }\n}\n```\n\nThis function uses the `qede_reuse_page` and `qede_rx_bd_ring_consume` functions to recycle the ring buffer. It does this by calling these functions a specified number of times.\n\nPlease note that the `qede_reuse_page` and `qede_rx_bd_ring_consume` functions are not defined in the provided code, so you'll need to implement them yourself.\n\nThe `qede_rx_bd_ring` function is designed to be used in a loop, so it can be used to recycle the ring buffer multiple times.\n\nThe `qede_recycle_rx_bd_ring` function is part of a larger system that manages a series of data packets, and it's used to recycle a specific ring buffer.\n",
    "logprobs": null,
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant that generates code. Given a code summary, provide the corresponding code implementation in C. warpper your code with ```.",
            "loss": null
        },
        {
            "role": "user",
            "content": "\n\nThe function `qede_recycle_rx_bd_ring` is used to recycle a ring buffer for a specific number of times. The ring buffer is a data structure that is used to store and manage a series of data packets. The function takes a pointer to a `qede_rx_queue` structure and a count of how many times the ring buffer should be recycled.\n\nThe function iterates the specified number of times, each time consuming a packet from the ring buffer and reusing it. The `qede_reuse_page` function is used to reuse the page of the ring buffer, and the `qede_rx_bd_ring_consume` function is used to consume a packet from the ring buffer.\n\nThe function is designed to be used in a loop, so it can be used to recycle the ring buffer multiple times.\n\nThe function is part of a larger system that manages a series of data packets, and it's used to recycle a specific ring buffer.\n",
            "loss": null
        },
        {
            "role": "assistant",
            "content": " \n\nHere is a simple implementation of the `qede_recycle_rx_bd_ring` function in C:\n\n```\nvoid qede_recycle_rx_bd_ring(struct qede_rx_queue *rxq, u8 count)\n{\n\tstruct sw_rx_data *curr_cons;\n\n\tfor (; count > 0; count--) {\n\t\tcurr_cons = &rxq->sw_rx_ring[rxq->sw_rx_cons & NUM_RX_BDS_MAX];\n\t\tqede_reuse_page(rxq, curr_cons);\n\t\tqede_rx_bd_ring_consume(rxq);\n\t}\n}\n```\n\nThis function uses the `qede_reuse_page` and `qede_rx_bd_ring_consume` functions to recycle the ring buffer. It does this by calling these functions a specified number of times.\n\nPlease note that the `qede_reuse_page` and `qede_rx_bd_ring_consume` functions are not defined in the provided code, so you'll need to implement them yourself.\n\nThe `qede_rx_bd_ring` function is designed to be used in a loop, so it can be used to recycle the ring buffer multiple times.\n\nThe `qede_recycle_rx_bd_ring` function is part of a larger system that manages a series of data packets, and it's used to recycle a specific ring buffer.\n",
            "loss": null
        }
    ]
}
2. 根据spmi值分别筛选出前10%和后10%的数据，并对每一部分采样20个样本
3. 将采样数据打乱顺序，问答对保存在/workspace/eval/experiments1/code_sum_pair/spmi/模型.json中，标签保存在label.json中
"""
import json
import os
import random
import re

root = "/workspace/eval/experiments1/spmi_result"

for model in os.listdir(root):
    path = os.path.join(root, model)
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()]
    data = sorted(data, key=lambda x: x['spmi'])
    # for item in data:
    #     print(item["messages"][1]['content'])
    #     input(item['spmi'])
    mean_spmi = sum(item['spmi'] for item in data) / len(data)
    print(f"model: {model}, mean_spmi: {mean_spmi}")
    n = len(data)
    top_10p = data[int(n*0.9):]
    bottom_10p = data[:int(n*0.1)]
    sampled_top = random.sample(top_10p, min(50, len(top_10p)))
    sampled_bottom = random.sample(bottom_10p, min(50, len(bottom_10p)))
    sampled = sampled_top + sampled_bottom
    random.shuffle(sampled)
    os.makedirs(f"/workspace/eval/experiments1/code_sum_pair/spmi/{model.split('.')[0]}", exist_ok=True)
    with open(f"/workspace/eval/experiments1/code_sum_pair/spmi/{model.split('.')[0]}/data.json", "w", encoding="utf-8") as f:
        for item in sampled:
            messages = [
                {
                    "role": "summary",
                    "content": item["messages"][1]['content'],
                },
                {
                    "role": "code",
                    "content": item['labels']
                }
            ]
            json.dump(messages, f)
            f.write("\n")
    with open(f"/workspace/eval/experiments1/code_sum_pair/spmi/{model.split('.')[0]}/label.json", "w", encoding="utf-8") as f:
        for item in sampled:
            json.dump(item['spmi'], f)
            f.write("\n")
