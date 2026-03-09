import json
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 配置路径
# =========================
CONFIG_PATH = "/home/data/students_user/2024/lr/code/MyProject/output/2/adapter/adapter_config.json"



OUTPUT_DIR = "rank_visualization_chart"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# =========================
# 读取配置
# =========================
with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

rank_pattern = cfg["rank_pattern"]
R = cfg.get("init_r", 32)  # 初始秩，通常为 32

# =========================
# 解析 rank_pattern
# =========================
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

layers = sorted(layer_module_rank.keys())
num_layers = len(layers)  # 应为 32

# =========================
# 计算每层的实际有效 rank
# =========================
q_ranks = np.zeros(num_layers, dtype=int)
v_ranks = np.zeros(num_layers, dtype=int)

for i, layer in enumerate(layers):
    q_ranks[i] = np.sum(layer_module_rank[layer]["q"])
    v_ranks[i] = np.sum(layer_module_rank[layer]["v"])

# =========================
# 绘图：实线 Q_proj，虚线 V_proj
# =========================
plt.figure(figsize=(14, 8))

# Q_proj：蓝色实线 + 圆点
plt.plot(layers, q_ranks, 
         marker='o', markersize=7, 
         linewidth=3, 
         label='Q_proj Effective Rank', 
         color='#1f77b4', 
         linestyle='-')  # 实线

# V_proj：橙色虚线 + 方点
plt.plot(layers, v_ranks, 
         marker='s', markersize=7, 
         linewidth=3, 
         label='V_proj Effective Rank', 
         color='#ff7f0e', 
         linestyle='--')  # 虚线

# 美化设置
plt.title("PS-AdaLoRA: Preference-Sensitive Effective Rank Allocation", fontsize=18, pad=20)
plt.xlabel("Transformer Layer", fontsize=14)
plt.ylabel("Effective Rank (0–32)", fontsize=14)
plt.grid(True, alpha=0.3, linestyle=':')
plt.legend(fontsize=13, loc='upper left')
plt.xticks(layers)
plt.ylim(0, R + 1)
plt.xlim(layers[0] - 0.5, layers[-1] + 0.5)

# 移除 fill_between，避免重合部分变色
# plt.fill_between(...)  # 已删除

plt.tight_layout()
# plt.savefig(f"{OUTPUT_DIR}/rank_allocation_line_chart_q_v_clean.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUTPUT_DIR}/rank_allocation_line_chart_q_v_clean.pdf", bbox_inches='tight')

plt.close()

print(f"Rank 分配折线图（实线Q + 虚线V）已保存至: {OUTPUT_DIR}/rank_allocation_line_chart_q_v_clean.png")