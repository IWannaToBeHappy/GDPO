# 统计spmi_result下的所有评测结果，并计算均值
import os
import jsonlines
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

root = "/workspace/eval/experiments2/spmi_result"
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
qwen3_base.jsonl:               avg_spmi=0.2744325627721846, count=1000
hunyuan_base.jsonl:             avg_spmi=0.2891525726318359, count=1000
llama_base.jsonl:               avg_spmi=0.37844319725036624, count=1000
glm_edge_base.jsonl:            avg_spmi=0.44430075073242187, count=1000
gemma_base.jsonl:               avg_spmi=0.46939395141601564, count=1000
deepseek_base.jsonl:            avg_spmi=0.5605736351013184, count=1000
nokl_deepseek_r2.jsonl:         avg_spmi=0.567017786026001, count=1000
arithmetic_deepseek_r2.jsonl:   avg_spmi=0.5680503215789795, count=1000
arithmetic_deepseek_r1.jsonl:   avg_spmi=0.5699897060394287, count=1000
geometric_deepseek_r1.jsonl:    avg_spmi=0.5705147571563721, count=1000
geometric_deepseek_r2.jsonl:    avg_spmi=0.5744383163452148, count=1000
"""

# 为所有文件分配确定的颜色
color_map = {
    "qwen3_base.jsonl":                "#7f7f7f",
    "hunyuan_base.jsonl":              "#bcbd22",
    "llama_base.jsonl":                "#2ca02c",
    "glm_edge_base.jsonl":             "#8c564b",
    "gemma_base.jsonl":                "#9467bd",
    "deepseek_base.jsonl":             "#393b79",
    "nokl_deepseek_r2.jsonl":          "#e377c2",
    "arithmetic_deepseek_r2.jsonl":    "#1f77b4",
    "arithmetic_deepseek_r1.jsonl":    "#17becf",
    "geometric_deepseek_r1.jsonl":     "#ff7f0e",
    "geometric_deepseek_r2.jsonl":     "#d62728",
}

show_root = "/workspace/eval/experiments2/show"
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
gemma_base.jsonl:               59.90% of spmi values are in [0.45, 0.55]
glm_edge_base.jsonl:            40.30% of spmi values are in [0.45, 0.55]
llama_base.jsonl:               26.30% of spmi values are in [0.45, 0.55]
qwen3_base.jsonl:               22.50% of spmi values are in [0.45, 0.55]
hunyuan_base.jsonl:             15.40% of spmi values are in [0.45, 0.55]
deepseek_base.jsonl:            36.90% of spmi values are in [0.45, 0.55]
geometric_deepseek_r1.jsonl:    28.90% of spmi values are in [0.45, 0.55]
arithmetic_deepseek_r2.jsonl:   28.70% of spmi values are in [0.45, 0.55]
geometric_deepseek_r2.jsonl:    26.80% of spmi values are in [0.45, 0.55]
arithmetic_deepseek_r1.jsonl:   26.80% of spmi values are in [0.45, 0.55]
nokl_deepseek_r2.jsonl:         26.40% of spmi values are in [0.45, 0.55]
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
qwen3_base.jsonl:               8.10% of spmi values are in [0.8, 1.0]
hunyuan_base.jsonl:             9.60% of spmi values are in [0.8, 1.0]
gemma_base.jsonl:               7.10% of spmi values are in [0.8, 1.0]
llama_base.jsonl:               11.30% of spmi values are in [0.8, 1.0]
glm_edge_base.jsonl:            14.40% of spmi values are in [0.8, 1.0]
deepseek_base.jsonl:            32.00% of spmi values are in [0.8, 1.0]
arithmetic_deepseek_r2.jsonl:   37.00% of spmi values are in [0.8, 1.0]
arithmetic_deepseek_r1.jsonl:   37.20% of spmi values are in [0.8, 1.0]
geometric_deepseek_r1.jsonl:    37.80% of spmi values are in [0.8, 1.0]
nokl_deepseek_r2.jsonl:         38.00% of spmi values are in [0.8, 1.0]
geometric_deepseek_r2.jsonl:    38.20% of spmi values are in [0.8, 1.0]
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
glm_edge_base.jsonl:            35.10% of spmi values are in [0.0, 0.4]
qwen3_base.jsonl:               62.40% of spmi values are in [0.0, 0.4]
gemma_base.jsonl:               20.20% of spmi values are in [0.0, 0.4]
llama_base.jsonl:               48.50% of spmi values are in [0.0, 0.4]
hunyuan_base.jsonl:             65.10% of spmi values are in [0.0, 0.4]
deepseek_base.jsonl:            23.10% of spmi values are in [0.0, 0.4]
geometric_deepseek_r1.jsonl:    26.90% of spmi values are in [0.0, 0.4]
arithmetic_deepseek_r1.jsonl:   27.10% of spmi values are in [0.0, 0.4]
arithmetic_deepseek_r2.jsonl:   27.10% of spmi values are in [0.0, 0.4]
geometric_deepseek_r2.jsonl:    27.50% of spmi values are in [0.0, 0.4]
nokl_deepseek_r2.jsonl:         28.10% of spmi values are in [0.0, 0.4]
"""

###############################
#           CDF
##############################

# 所有基线模型
for file, spmi_list in meta.items():
    if "ccsd" in file or "r1" in file or "r2" in file:
        continue
    if not spmi_list:
        print(f"Warning: {file} has no spmi values, skipping")
        continue
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
plt.savefig(os.path.join(show_root, "baseline_spmi_cdf.png"), dpi=200)
plt.clf()

# deepseek系列
for file, spmi_list in meta.items():
    if "deepseek" not in file:
        continue
    if not spmi_list:
        print(f"Warning: {file} has no spmi values, skipping")
        continue
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
plt.savefig(os.path.join(show_root, "deepseek_spmi_cdf.png"), dpi=200)
plt.clf()

###############################
#           CDF delta
##############################

# 1. 先计算基线的CDF
baseline_file = "deepseek_base.jsonl"
baseline_spmi_list = meta.get(baseline_file)
baseline_cdf = None
x_vals = np.linspace(0, 1.0, 501)

if baseline_spmi_list:
    sorted_vals = np.sort(np.array(baseline_spmi_list))
    n = len(sorted_vals)
    baseline_cdf = (np.searchsorted(sorted_vals, x_vals, side='right') / n) * 100.0
else:
    print(f"Warning: Baseline file '{baseline_file}' not found or is empty. Cannot compute delta.")

# 2. 计算其他deepseek模型CDF与基线的差值并绘图
if baseline_cdf is not None:
    for file, spmi_list in meta.items():
        # 跳过基线文件本身
        if file == baseline_file:
            continue
        
        if "deepseek" in file and "ccsd" not in file:
            if not spmi_list:
                print(f"Warning: {file} has no spmi values, skipping")
                continue
            
            # 计算当前模型的CDF
            sorted_vals = np.sort(np.array(spmi_list))
            n = len(sorted_vals)
            model_cdf = (np.searchsorted(sorted_vals, x_vals, side='right') / n) * 100.0
            
            # 计算与基线的差值
            delta_cdf = model_cdf - baseline_cdf
            
            # 绘制差值曲线
            plt.plot(x_vals, delta_cdf, label=f"{file[:-6]} vs {baseline_file[:-6]}", color=color_map.get(file))

    # 绘制参考零线
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    
    # 绘制图形
    plt.xlabel("spmi")
    plt.ylabel("CDF Delta (%)")
    plt.legend(loc='best', fontsize='small')
    os.makedirs(show_root, exist_ok=True)
    plt.savefig(os.path.join(show_root, "deepseek_spmi_cdf_delta.png"), dpi=200)
    plt.clf()
