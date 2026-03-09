import json
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

# =========================
# 配置路径
# =========================
CONFIG_PATH = "/home/data/students_user/2024/lr/code/MyProject/output/14/adapter/adapter_config.json"
OUTPUT_DIR = "rank_visualization_gray"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# =========================
# 读取配置
# =========================
with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

rank_pattern = cfg["rank_pattern"]
R = cfg.get("init_r", 32)

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
        layer_module_rank[layer]["q"] = np.array(mask, dtype=float)
    elif "v_proj" in name:
        layer_module_rank[layer]["v"] = np.array(mask, dtype=float)

layers = sorted(layer_module_rank.keys())
num_layers = len(layers)

# =========================
# 构造矩阵
# =========================
q_matrix = np.zeros((num_layers, R))
v_matrix = np.zeros((num_layers, R))
for i, layer in enumerate(layers):
    q_matrix[i] = layer_module_rank[layer].get("q", 0)
    v_matrix[i] = layer_module_rank[layer].get("v", 0)

merged_matrix = q_matrix + v_matrix  # [0, 1, 2]

# =========================
# 图 1：Merged Q+V（灰白黑，秩越高越黑）
# =========================
plt.figure(figsize=(12, 8))
im = plt.imshow(merged_matrix, aspect="auto", cmap="gray_r", vmin=0, vmax=2, interpolation="nearest")
plt.colorbar(im, label="Active Ranks (Q + V)", ticks=[0, 1, 2])
plt.title("Merged Q + V Rank Allocation\n(Darker = Higher Rank)", fontsize=16, pad=20)
plt.xlabel("Rank Index (0–31)")
plt.ylabel("Transformer Layer (0–31)")
plt.yticks(range(num_layers), layers)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/rank_allocation_heatmap_gray.png", dpi=300, bbox_inches='tight')
plt.close()

# =========================
# 图 2：Layer-wise Effective Rank
# =========================
effective_rank = merged_matrix.sum(axis=1)
mean_rank = effective_rank.mean()

plt.figure(figsize=(10, 5))
plt.plot(layers, effective_rank, marker="o", markersize=5, linewidth=2, color="black")
plt.axhline(mean_rank, color="gray", linestyle="--", label=f"Mean = {mean_rank:.2f}")
plt.title("Layer-wise Effective Rank (Q + V)", fontsize=16)
plt.xlabel("Transformer Layer")
plt.ylabel("Effective Rank")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/effective_rank_per_layer_gray.png", dpi=300, bbox_inches='tight')
plt.close()

# =========================
# 图 3：Q/V 分开热力图（灰白黑风格，秩越高越黑）
# =========================
fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True, constrained_layout=True)

# 灰白黑反转色图：值越大越黑
fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True, constrained_layout=True)

# 灰白黑反转色图：值越大越黑
cmap = "gray_r"   # 0=白（pruned），1=黑（active）

im_q = axes[0].imshow(q_matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
axes[0].set_title("Q-Projection Rank Allocation\n(Darker = More Active Ranks)", fontsize=15, pad=15)
axes[0].set_xlabel("Rank Index (0–31)")
axes[0].set_ylabel("Transformer Layer (0–31)")
axes[0].set_yticks(range(num_layers))
axes[0].set_yticklabels(layers)

im_v = axes[1].imshow(v_matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
axes[1].set_title("V-Projection Rank Allocation\n(Darker = More Active Ranks)", fontsize=15, pad=15)
axes[1].set_xlabel("Rank Index (0–31)")

# 共享 colorbar
cbar = fig.colorbar(im_v, ax=axes, location='right', shrink=0.8, pad=0.05)
cbar.set_label("Rank Activity\n(White = Pruned, Black = Active)", fontsize=12)
cbar.set_ticks([0, 0.5, 1])

# 修改主标题：居中显示，只保留“AdaLoRA Rank Allocation”
fig.suptitle("AdaLoRA Rank Allocation              ", fontsize=18, y=0.98, ha='center')

plt.savefig(f"{OUTPUT_DIR}/qv_rank_allocation_heatmap_gray.png", dpi=300, bbox_inches='tight')
plt.close()

# =========================
# 新图：层级密度条形热力图（颜色连续丰富）
plt.figure(figsize=(12, 8))

# 每层 effective rank 归一化到 [0,1] 用于颜色
norm_effective = effective_rank / effective_rank.max()

# 创建一个伪矩阵：每行重复同一值（制造“条形”效果）
density_matrix = np.repeat(norm_effective[:, np.newaxis], R, axis=1)

im = plt.imshow(density_matrix, aspect="auto", cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
plt.colorbar(im, label="Normalized Effective Rank (0=low, 1=high)")
plt.title("Layer-wise Rank Density Heatmap\n(Darker = Higher Total Active Ranks)", fontsize=16, pad=20)
plt.xlabel("Rank Index (0–31) - Uniform Fill for Visualization")
plt.ylabel("Transformer Layer (0–31)")
plt.yticks(range(num_layers), layers)

# 可选：在每行右侧标注具体数字
for i, layer in enumerate(layers):
    plt.text(R + 1, i, f"{int(effective_rank[i])}", va='center', ha='left', color='black', fontsize=8)

plt.xlim(0, R + 10)  # 留空间给文字
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/rank_density_continuous_gray.png", dpi=300, bbox_inches='tight')
plt.close()
# 统计信息
# =========================
print("===== Rank Allocation Statistics (Gray Scale Version) =====")
print(f"Total Layers       : {num_layers}")
print(f"Initial Rank Space : {R}")
print(f"Mean Effective Rank (Q+V): {mean_rank:.2f}")
print(f"Min / Max Effective Rank : {effective_rank.min():.2f} / {effective_rank.max():.2f}")

top_layers = sorted(zip(layers, effective_rank), key=lambda x: -x[1])[:5]
print("\nTop-5 Layers (Highest Rank Allocation):")
for l, r in top_layers:
    print(f"  Layer {l:02d} : {r:.2f}")

print(f"\nGray-scale figures saved to: {OUTPUT_DIR}/")