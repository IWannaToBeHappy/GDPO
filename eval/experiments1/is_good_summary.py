# 我们使用deepseek API模拟专家进行摘要质量判定，API的能力远胜于1b小模型，可以充当教师模型
# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI
import json

client = OpenAI(
    api_key='',
    base_url="https://api.deepseek.com")

def is_good_summary(text):
    prompt = f"""
You are an expert in evaluating the quality of code summaries. Given a code and its corresponding summary. 判断他能否有助于还原代码？如果是，返回True，如果不是，返回False。请严格按照True或者False的格式返回结果，不要添加任何多余的内容。
{text}
"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )

    content = response.choices[0].message.content
    if "true" in content.lower() and "false" not in content.lower():
        return True
    elif "false" in content.lower() and "true" not in content.lower():
        return False
    else:
        result = input(text+"\n\n"+content)
        if not result.strip(): #什么都不输入，返回True
            return True
        return False # 任意输入返回False
    
# 评估bleu:
# bleu_root = "/workspace/eval/experiments1/code_sum_pair/bleu"
# for model in os.listdir(bleu_root):
#     for dataset in ["general_qa_ccsd_c_functions_all_data.jsonl","general_qa_tl_codesum_test.jsonl"]:
#         ccsd = os.path.join(bleu_root, model, dataset+".json")
#         ccsd_label = os.path.join(bleu_root, model, dataset+"_label.json")
#         with open(ccsd_label, "r") as f:
#             labels = json.load(f)
#         with open(ccsd, "r") as f:
#             lines = f.readlines()
#         tp,fp,fn,tn = 0,0,0,0
#         for line in lines:
#             j = json.loads(line)
#             index = j["index"]
#             if index in labels:
#                 is_good = True
#             else:
#                 is_good = False
#             ds = is_good_summary(line)
#             if is_good and ds:
#                 tp += 1
#             elif not is_good and ds:
#                 fp += 1
#             elif is_good and not ds:
#                 fn += 1
#             else:
#                 tn += 1
#         # 计算F1
#         precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#         recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#         f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
#         print(f"BLEU Model: {model},Dataset:{dataset} TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
"""
BLEU Model: qwen3-1b,Dataset:general_qa_ccsd_c_functions_all_data.jsonl                         TP: 46, FP: 39, FN: 4, TN: 11, Precision: 0.5412, Recall: 0.9200, F1: 0.6815
BLEU Model: qwen3-1b,Dataset:general_qa_tl_codesum_test.jsonl                                   TP: 44, FP: 32, FN: 6, TN: 18, Precision: 0.5789, Recall: 0.8800, F1: 0.6984
                                                                                                    90      71      10      29                                          0.6429
BLEU Model: hunyuan-1_8b-instruct,Dataset:general_qa_ccsd_c_functions_all_data.jsonl            TP: 29, FP: 18, FN: 21, TN: 32, Precision: 0.6170, Recall: 0.5800, F1: 0.5979
BLEU Model: hunyuan-1_8b-instruct,Dataset:general_qa_tl_codesum_test.jsonl                      TP: 35, FP: 22, FN: 15, TN: 28, Precision: 0.6140, Recall: 0.7000, F1: 0.6542
                                                                                                    64      40      36      60                                          0.5614
BLEU Model: deepseek-coder-1_3b-instruct,Dataset:general_qa_ccsd_c_functions_all_data.jsonl     TP: 18, FP: 7, FN: 32, TN: 43, Precision: 0.7200, Recall: 0.3600, F1: 0.4800
BLEU Model: deepseek-coder-1_3b-instruct,Dataset:general_qa_tl_codesum_test.jsonl               TP: 16, FP: 0, FN: 34, TN: 50, Precision: 1.0000, Recall: 0.3200, F1: 0.4848
                                                                                                    34      7       66      93                                          0.4048
BLEU Model: llama-3_2-1B-instruct,Dataset:general_qa_ccsd_c_functions_all_data.jsonl            TP: 2, FP: 0, FN: 48, TN: 50, Precision: 1.0000, Recall: 0.0400, F1: 0.0769
BLEU Model: llama-3_2-1B-instruct,Dataset:general_qa_tl_codesum_test.jsonl                      TP: 8, FP: 0, FN: 42, TN: 50, Precision: 1.0000, Recall: 0.1600, F1: 0.2759
                                                                                                    10      0       90      100                                         0.1667
BLEU Model: glm-edge-1_5b-chat,Dataset:general_qa_ccsd_c_functions_all_data.jsonl               TP: 35, FP: 29, FN: 15, TN: 21, Precision: 0.5469, Recall: 0.7000, F1: 0.6140
BLEU Model: glm-edge-1_5b-chat,Dataset:general_qa_tl_codesum_test.jsonl                         TP: 27, FP: 22, FN: 23, TN: 28, Precision: 0.5510, Recall: 0.5400, F1: 0.5455
                                                                                                    62      51      38      49                                          0.5536
BLEU Model: gemma-2-2b-it,Dataset:general_qa_ccsd_c_functions_all_data.jsonl                    TP: 36, FP: 32, FN: 14, TN: 18, Precision: 0.5294, Recall: 0.7200, F1: 0.6102
BLEU Model: gemma-2-2b-it,Dataset:general_qa_tl_codesum_test.jsonl                              TP: 38, FP: 31, FN: 12, TN: 19, Precision: 0.5507, Recall: 0.7600, F1: 0.6387
                                                                                                    74      63      26      37                                          0.5968
BLEU Model: phi-4-mini-instruct,Dataset:general_qa_ccsd_c_functions_all_data.jsonl              TP: 39, FP: 13, FN: 11, TN: 37, Precision: 0.7500, Recall: 0.7800, F1: 0.7647
BLEU Model: phi-4-mini-instruct,Dataset:general_qa_tl_codesum_test.jsonl                        TP: 38, FP: 13, FN: 12, TN: 37, Precision: 0.7451, Recall: 0.7600, F1: 0.7525
                                                                                                    77      26      23      74
Total: TP = 411 FP = 258 FN = 289 TN = 442 F1 = 0.6004
"""


# 评估rouge:
# rouge_root = "/workspace/eval/experiments1/code_sum_pair/rouge"
# for model in os.listdir(rouge_root):
#     for dataset in ["general_qa_ccsd_c_functions_all_data.jsonl","general_qa_tl_codesum_test.jsonl"]:
#         ccsd = os.path.join(rouge_root, model, dataset+".json")
#         ccsd_label = os.path.join(rouge_root, model, dataset+"_label.json")
#         with open(ccsd_label, "r") as f:
#             labels = json.load(f)
#         with open(ccsd, "r") as f:
#             lines = f.readlines()
#         tp,fp,fn,tn = 0,0,0,0
#         for line in lines:
#             j = json.loads(line)
#             index = j["index"]
#             if index in labels:
#                 is_good = True
#             else:
#                 is_good = False
#             ds = is_good_summary(line)
#             if is_good and ds:
#                 tp += 1
#             elif not is_good and ds:
#                 fp += 1
#             elif is_good and not ds:
#                 fn += 1
#             else:
#                 tn += 1
#         # 计算F1
#         precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#         recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#         f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
#         print(f"rouge Model: {model},Dataset:{dataset} TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
"""
rouge Model: qwen3-1b,Dataset:general_qa_ccsd_c_functions_all_data.jsonl                        TP: 45, FP: 44, FN: 5, TN: 6, Precision: 0.5056, Recall: 0.9000, F1: 0.6475
rouge Model: qwen3-1b,Dataset:general_qa_tl_codesum_test.jsonl                                  TP: 46, FP: 38, FN: 4, TN: 12, Precision: 0.5476, Recall: 0.9200, F1: 0.6866
                                                                                                    91      82      9       18                                          0.6454
rouge Model: hunyuan-1_8b-instruct,Dataset:general_qa_ccsd_c_functions_all_data.jsonl           TP: 38, FP: 15, FN: 12, TN: 35, Precision: 0.7170, Recall: 0.7600, F1: 0.7379
rouge Model: hunyuan-1_8b-instruct,Dataset:general_qa_tl_codesum_test.jsonl                     TP: 37, FP: 27, FN: 13, TN: 23, Precision: 0.5781, Recall: 0.7400, F1: 0.6491
                                                                                                    75      42      25      58                                          0.6000
rouge Model: deepseek-coder-1_3b-instruct,Dataset:general_qa_ccsd_c_functions_all_data.jsonl    TP: 22, FP: 5, FN: 28, TN: 45, Precision: 0.8148, Recall: 0.4400, F1: 0.5714
rouge Model: deepseek-coder-1_3b-instruct,Dataset:general_qa_tl_codesum_test.jsonl              TP: 20, FP: 3, FN: 30, TN: 47, Precision: 0.8696, Recall: 0.4000, F1: 0.5479
                                                                                                    42      8       58      92                                          0.4565
rouge Model: llama-3_2-1B-instruct,Dataset:general_qa_ccsd_c_functions_all_data.jsonl           TP: 1, FP: 0, FN: 49, TN: 50, Precision: 1.0000, Recall: 0.0200, F1: 0.0392
rouge Model: llama-3_2-1B-instruct,Dataset:general_qa_tl_codesum_test.jsonl                     TP: 3, FP: 0, FN: 47, TN: 50, Precision: 1.0000, Recall: 0.0600, F1: 0.1132
                                                                                                    4       0       96      100                                         0.0741
rouge Model: glm-edge-1_5b-chat,Dataset:general_qa_ccsd_c_functions_all_data.jsonl              TP: 35, FP: 26, FN: 15, TN: 24, Precision: 0.5738, Recall: 0.7000, F1: 0.6306
rouge Model: glm-edge-1_5b-chat,Dataset:general_qa_tl_codesum_test.jsonl                        TP: 31, FP: 33, FN: 19, TN: 17, Precision: 0.4844, Recall: 0.6200, F1: 0.5439
                                                                                                    66      59      34      41                                          0.5690
rouge Model: gemma-2-2b-it,Dataset:general_qa_ccsd_c_functions_all_data.jsonl                   TP: 38, FP: 36, FN: 12, TN: 14, Precision: 0.5135, Recall: 0.7600, F1: 0.6129
rouge Model: gemma-2-2b-it,Dataset:general_qa_tl_codesum_test.jsonl                             TP: 36, FP: 34, FN: 14, TN: 16, Precision: 0.5143, Recall: 0.7200, F1: 0.6000
                                                                                                    74      70      26      30                                        0.5968
rouge Model: phi-4-mini-instruct,Dataset:general_qa_ccsd_c_functions_all_data.jsonl             TP: 36, FP: 21, FN: 14, TN: 29, Precision: 0.6316, Recall: 0.7200, F1: 0.6729
rouge Model: phi-4-mini-instruct,Dataset:general_qa_tl_codesum_test.jsonl                       TP: 41, FP: 11, FN: 9, TN: 39, Precision: 0.7885, Recall: 0.8200, F1: 0.8039
Total: TP = 429 FP = 293 FN = 304 TN = 270 F1: 0.5897
"""
# 评估spmi
# spmi_root = "/workspace/eval/experiments1/code_sum_pair/spmi"
# for model in os.listdir(spmi_root):
#     data_path = os.path.join(spmi_root, model, "data.json")
#     label_path = os.path.join(spmi_root, model, "label.json")
#     with open(label_path, "r") as f:
#         labels = [float(line.strip()) for line in f.readlines()]
#     # 前一半是好样本,总计100个样本
#     thersholds = sorted(labels.copy())[50]
#     with open(data_path, "r") as f:
#         lines = f.readlines()
#     tp,fp,fn,tn = 0,0,0,0
#     for i, (line,spmi) in enumerate(zip(lines,labels)):
#         if spmi >= thersholds:
#             is_good = True
#         else:
#             is_good = False
#         ds = is_good_summary(line)
#         if is_good and ds:
#             tp += 1
#         elif not is_good and ds:
#             fp += 1
#         elif is_good and not ds:
#             fn += 1
#         else:
#             tn += 1
#     # 计算F1
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#     f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
#     print(f"SPMI Model: {model} TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

"""
前后分开：
SPMI Model: qwen3-1b                        TP: 49, FP: 49, FN: 1, TN: 1, Precision: 0.5000, Recall: 0.9800, F1: 0.6622
SPMI Model: hunyuan-1_8b-instruct           TP: 49, FP: 47, FN: 1, TN: 3, Precision: 0.5104, Recall: 0.9800, F1: 0.6712
SPMI Model: deepseek-coder-1_3b-instruct    TP: 44, FP: 33, FN: 6, TN: 17, Precision: 0.5714, Recall: 0.8800, F1: 0.6929
SPMI Model: llama-3_2-1B-instruct           TP: 36, FP: 19, FN: 14, TN: 31, Precision: 0.6545, Recall: 0.7200, F1: 0.6857
SPMI Model: glm-edge-1_5b-chat              TP: 47, FP: 49, FN: 3, TN: 1, Precision: 0.4896, Recall: 0.9400, F1: 0.6438
SPMI Model: gemma-2-2b-it                   TP: 48, FP: 49, FN: 2, TN: 1, Precision: 0.4948, Recall: 0.9600, F1: 0.6531
SPMI Model: phi-4-mini-instruct             TP: 45, FP: 49, FN: 5, TN: 1, Precision: 0.4787, Recall: 0.9000, F1: 0.6250
Total: TP 318 FP 295 FN 32 TN 55, Precision: 0.5183, Recall: 0.9082, F1: 0.6619
"""

spmi_root = "/workspace/eval/experiments1/code_sum_pair/spmi"
for model in os.listdir(spmi_root):
    data_path = os.path.join(spmi_root, model, "data.json")
    label_path = os.path.join(spmi_root, model, "label.json")
    with open(label_path, "r") as f:
        labels = [float(line.strip()) for line in f.readlines()]
    with open(data_path, "r") as f:
        lines = f.readlines()
    tp,fp,fn,tn = 0,0,0,0
    for i, (line,spmi) in enumerate(zip(lines,labels)):
        if spmi >= 0.5:
            is_good = True
        else:
            is_good = False
        ds = is_good_summary(line)
        if is_good and ds:
            tp += 1
        elif not is_good and ds:
            fp += 1
        elif is_good and not ds:
            fn += 1
        else:
            tn += 1
    # 计算F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"SPMI Model: {model} TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
"""
0.5分好坏：
SPMI Model: qwen3-1b                        TP: 49, FP: 49, FN: 1, TN: 1, Precision: 0.5000, Recall: 0.9800, F1: 0.6622
SPMI Model: hunyuan-1_8b-instruct           TP: 95, FP: 0, FN: 5, TN: 0, Precision: 1.0000, Recall: 0.9500, F1: 0.9744
SPMI Model: deepseek-coder-1_3b-instruct    TP: 56, FP: 21, FN: 11, TN: 12, Precision: 0.7273, Recall: 0.8358, F1: 0.7778
SPMI Model: llama-3_2-1B-instruct           TP: 34, FP: 18, FN: 16, TN: 32, Precision: 0.6538, Recall: 0.6800, F1: 0.6667
SPMI Model: glm-edge-1_5b-chat              TP: 91, FP: 5, FN: 3, TN: 1, Precision: 0.9479, Recall: 0.9681, F1: 0.9579
SPMI Model: gemma-2-2b-it                   TP: 96, FP: 1, FN: 3, TN: 0, Precision: 0.9897, Recall: 0.9697, F1: 0.9796
SPMI Model: phi-4-mini-instruct             TP: 44, FP: 49, FN: 6, TN: 1, Precision: 0.4731, Recall: 0.8800, F1: 0.6154
Total: TP 465 FP 143 FN 45 TN 47, Precision: 0.7648, Recall: 0.9118, F1: 0.8319
"""