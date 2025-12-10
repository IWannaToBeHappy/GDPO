# 统计spmi_result下的所有评测结果，并计算均值
import os
import jsonlines
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

root = "/workspace/eval/experiments3/spmi_result"
meta = {}
for file in os.listdir(root):
    if not file.endswith(".jsonl"):
        continue
    file_path = os.path.join(root, file)
    spmi_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for obj in jsonlines.Reader(f):
            if "spmi" in obj:
                spmi_list.append(obj["spmi"])
    avg_spmi = sum(spmi_list) / len(spmi_list) if spmi_list else 0.0
    print(f"{file}: avg_spmi={avg_spmi}, count={len(spmi_list)}")
    meta[file] = spmi_list
"""
promptcs.jsonl: avg_spmi=0.5446258185339001, count=5183
pt4code.jsonl:  avg_spmi=0.5456620631349575, count=5183
gdpo.jsonl:     avg_spmi=0.5475462067461878, count=5183
"""

# 为每个文件预先分配确定的颜色（按文件名排序，保证不同图中颜色一致）
color_map = {
    "gdpo.jsonl":               "#d62728",
    "promptcs.jsonl":           "#0e3aff",
    "pt4code.jsonl":            "#2ca02c",
}

show_root = "/workspace/eval/experiments3/show"
if not os.path.exists(show_root):
    os.makedirs(show_root)
import numpy as np


#################################
#       数值统计
##################################
# 0.5+-0.05范围内的比例
for file, spmi_list in meta.items():
    if not spmi_list:
        print(f"Warning: {file} has no spmi values, skipping")
        continue
    count_in_range = sum(1 for v in spmi_list if 0.45 <= v <= 0.55)
    ratio_in_range = count_in_range / len(spmi_list) * 100.0
    print(f"{file}: {ratio_in_range:.2f}% of spmi values are in [0.45, 0.55]")
"""
promptcs.jsonl: 32.90% of spmi values are in [0.45, 0.55]
pt4code.jsonl:  32.43% of spmi values are in [0.45, 0.55]
gdpo.jsonl:     22.46% of spmi values are in [0.45, 0.55]
"""

# 0.8-1.0范围内的比例
for file, spmi_list in meta.items():
    if not spmi_list:
        print(f"Warning: {file} has no spmi values, skipping")
        continue
    count_in_range = sum(1 for v in spmi_list if 0.8 <= v <= 1.0)
    ratio_in_range = count_in_range / len(spmi_list) * 100.0
    print(f"{file}: {ratio_in_range:.2f}% of spmi values are in [0.8, 1.0]")
"""
promptcs.jsonl: 31.18% of spmi values are in [0.8, 1.0]
pt4code.jsonl:  31.45% of spmi values are in [0.8, 1.0]
gdpo.jsonl:     35.50% of spmi values are in [0.8, 1.0]
"""

# 0-0.4范围内的比例
for file, spmi_list in meta.items():
    if not spmi_list:
        print(f"Warning: {file} has no spmi values, skipping")
        continue
    count_in_range = sum(1 for v in spmi_list if 0.0 <= v <= 0.4)
    ratio_in_range = count_in_range / len(spmi_list) * 100.0
    print(f"{file}: {ratio_in_range:.2f}% of spmi values are in [0.0, 0.4]")
"""
promptcs.jsonl: 25.68% of spmi values are in [0.0, 0.4]
pt4code.jsonl:  26.07% of spmi values are in [0.0, 0.4]
gdpo.jsonl:     31.26% of spmi values are in [0.0, 0.4]
"""

###############################
#           CDF
##############################

# 所有基线模型
for file, spmi_list in meta.items():
    # 使用经验分布函数（ECDF）: 排序并计算小于等于每个阈值的比例
    sorted_vals = np.sort(np.array(spmi_list))
    n = len(sorted_vals)
    # x 为 sorted_vals 的扩展，y 为对应的累计百分比
    # 为了让折线在 0 到 1 上平滑展示，我们在固定的点上插值
    x_vals = np.linspace(0, 1.0, 501)
    # 对于每个 x，计算比例 (<= x)
    y_vals = (np.searchsorted(sorted_vals, x_vals, side='right') / n )* 100.0
    plt.plot(x_vals, y_vals, label=file[:-6], color=color_map.get(file))
plt.xlabel("spmi")
plt.ylabel("Cumulative Frequency (%)")
# plt.title("SPMI CDF for Baseline Models")
plt.legend(loc='best', fontsize='small')
os.makedirs(show_root, exist_ok=True)
plt.savefig(os.path.join(show_root, "spmi_cdf.png"), dpi=200)
plt.clf()

