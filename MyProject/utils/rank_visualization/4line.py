import json
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 配置路径
# =========================
CONFIG_PATH = "/home/data/students_user/2024/lr/code/MyProject/output/2/adapter/adapter_config.json"
CONFIG_PATH_ps = "/home/data/students_user/2024/lr/code/MyProject/output/20/adapter/adapter_config.json"

OUTPUT_DIR = "rank_visualization_chart"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# =========================
# 读取两个配置
# =========================
def load_config(config_path):
    with open(config_path, "r") as f:
        cfg = json.load(f)
    
    rank_pattern = cfg["rank_pattern"]
    R = cfg.get("init_r", 32)  # 初始秩，通常为 32
    
    # 解析 rank_pattern
    layer_module_rank = {}
    
    for name, mask in rank_pattern.items():
        m = re.search(r"model\.layers\.(\d+)\.", name)
        if not m:
            continue
        layer = int(m.group(1))
        
        if layer not in layer_module_rank:
            layer_module_rank[layer] = {"q": np.zeros(R), "v": np.zeros(R)}
        
        if "q_proj" in name:
            layer_module_rank[layer]["q"] = np.array(mask, dtype=bool)
        elif "v_proj" in name:
            layer_module_rank[layer]["v"] = np.array(mask, dtype=bool)
    
    return layer_module_rank, R

# 加载两个配置
layer_module_rank1, R1 = load_config(CONFIG_PATH)  # 无PS
layer_module_rank2, R2 = load_config(CONFIG_PATH_ps)  # 有PS

# 确保R相同
R = R1 if R1 == R2 else 32

# =========================
# 计算每层的实际有效 rank
# =========================
def calculate_ranks(layer_module_rank):
    layers = sorted(layer_module_rank.keys())
    num_layers = len(layers)
    
    q_ranks = np.zeros(num_layers, dtype=int)
    v_ranks = np.zeros(num_layers, dtype=int)
    
    for i, layer in enumerate(layers):
        q_ranks[i] = np.sum(layer_module_rank[layer]["q"])
        v_ranks[i] = np.sum(layer_module_rank[layer]["v"])
    
    return layers, q_ranks, v_ranks

# 计算两组数据
layers1, q_ranks1, v_ranks1 = calculate_ranks(layer_module_rank1)
layers2, q_ranks2, v_ranks2 = calculate_ranks(layer_module_rank2)

# 确保层数一致，如果不一致则取并集
all_layers = sorted(set(layers1) | set(layers2))

# 创建完整的层列表
full_layers = list(range(min(all_layers), max(all_layers) + 1))

# 创建函数来填充缺失的层（用0填充）
def fill_missing_ranks(layers, ranks, full_layers):
    rank_dict = dict(zip(layers, ranks))
    filled_ranks = []
    for layer in full_layers:
        if layer in rank_dict:
            filled_ranks.append(rank_dict[layer])
        else:
            filled_ranks.append(0)
    return np.array(filled_ranks)

# 填充所有层的秩
q_ranks1_full = fill_missing_ranks(layers1, q_ranks1, full_layers)
v_ranks1_full = fill_missing_ranks(layers1, v_ranks1, full_layers)
q_ranks2_full = fill_missing_ranks(layers2, q_ranks2, full_layers)
v_ranks2_full = fill_missing_ranks(layers2, v_ranks2, full_layers)

# =========================
# 绘图：四个曲线在一张图上
# =========================
plt.figure(figsize=(16, 10))

# 1. Q_proj (无PS) - 蓝色虚线
plt.plot(full_layers, q_ranks1_full, 
         marker='o', markersize=6, 
         linewidth=2.5, 
         label='Q_proj (w/o PS)', 
         color='#1f77b4',  # 蓝色
         linestyle='--',   # 虚线
         alpha=0.8)

# 2. Q_proj (有PS) - 蓝色实线
plt.plot(full_layers, q_ranks2_full, 
         marker='o', markersize=6, 
         linewidth=2.5, 
         label='Q_proj (w/ PS)', 
         color='#1f77b4',  # 蓝色
         linestyle='-',    # 实线
         alpha=0.9)

# 3. V_proj (无PS) - 橙色虚线
plt.plot(full_layers, v_ranks1_full, 
         marker='s', markersize=6, 
         linewidth=2.5, 
         label='V_proj (w/o PS)', 
         color='#ff7f0e',  # 橙色
         linestyle='--',   # 虚线
         alpha=0.8)

# 4. V_proj (有PS) - 橙色实线
plt.plot(full_layers, v_ranks2_full, 
         marker='s', markersize=6, 
         linewidth=2.5, 
         label='V_proj (w/ PS)', 
         color='#ff7f0e',  # 橙色
         linestyle='-',    # 实线
         alpha=0.9)

# 美化设置
plt.title("AdaLoRA Effective Rank Allocation per Layer\nComparison with/without PS", fontsize=18, pad=20)
plt.xlabel("Transformer Layer", fontsize=14)
plt.ylabel("Effective Rank (0–32)", fontsize=14)
plt.grid(True, alpha=0.2, linestyle=':')

# 改进图例显示，分组显示
from matplotlib.lines import Line2D

# 创建自定义图例句柄
legend_elements = [
    Line2D([0], [0], color='#1f77b4', lw=2.5, linestyle='--', label='Q_proj (w/o PS)'),
    Line2D([0], [0], color='#1f77b4', lw=2.5, linestyle='-', label='Q_proj (w/ PS)'),
    Line2D([0], [0], color='#ff7f0e', lw=2.5, linestyle='--', label='V_proj (w/o PS)'),
    Line2D([0], [0], color='#ff7f0e', lw=2.5, linestyle='-', label='V_proj (w/ PS)'),
]

plt.legend(handles=legend_elements, fontsize=12, loc='upper left', framealpha=0.9)

# 设置坐标轴
if len(full_layers) > 10:
    # 如果层数多，只显示部分刻度
    step = max(1, len(full_layers) // 10)
    plt.xticks(full_layers[::step])
else:
    plt.xticks(full_layers)

plt.ylim(0, R + 1)
plt.xlim(full_layers[0] - 0.5, full_layers[-1] + 0.5)

# 添加网格线
plt.axhline(y=R/2, color='gray', linestyle=':', alpha=0.3, linewidth=1)

plt.tight_layout()

# 保存图片
output_path = f"{OUTPUT_DIR}/rank_allocation_comparison_ps_vs_nops.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"四曲线对比图已保存至: {output_path}")
print(f"配置1 (无PS): {CONFIG_PATH}")
print(f"配置2 (有PS): {CONFIG_PATH_ps}")
print(f"总层数: {len(full_layers)}")
print(f"Q_proj (无PS) 平均秩: {np.mean(q_ranks1_full):.2f}")
print(f"Q_proj (有PS) 平均秩: {np.mean(q_ranks2_full):.2f}")
print(f"V_proj (无PS) 平均秩: {np.mean(v_ranks1_full):.2f}")
print(f"V_proj (有PS) 平均秩: {np.mean(v_ranks2_full):.2f}")