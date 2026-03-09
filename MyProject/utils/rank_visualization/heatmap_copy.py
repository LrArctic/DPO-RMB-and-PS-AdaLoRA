import json
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

# =========================
# 0. 输出目录
# =========================
OUT_DIR = Path("heatmap")
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "adalora_rank_heatmap.png"

# =========================
# 1. 读取 adapter_config.json  2:0    23:0.001  14:0.002
# =========================
CONFIG_PATH = "/home/data/students_user/2024/lr/code/MyProject/output/2/adapter/adapter_config.json"

with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

rank_pattern = cfg["rank_pattern"]
init_r = cfg["init_r"]

# =========================
# 2. 统计每层每模块的 r
# =========================
layer_ranks = defaultdict(lambda: defaultdict(int))
layer_re = re.compile(r"model\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)\.lora_E")

for name, mask in rank_pattern.items():
    m = layer_re.match(name)
    if not m:
        continue
    layer_id = int(m.group(1))
    module = m.group(2)
    layer_ranks[layer_id][module] = sum(mask)

# =========================
# 3. 构造 heatmap
# =========================
num_layers = max(layer_ranks.keys()) + 1
modules = ["q_proj", "v_proj"]

heatmap = np.zeros((len(modules), num_layers))  # 注意这里直接定义成 (2, L)

for j, m in enumerate(modules):
    for layer in range(num_layers):
        heatmap[j, layer] = layer_ranks[layer].get(m, 0)

# =========================
# 4. 绘制 & 保存（白 → 黑）
# =========================
plt.figure(figsize=(12, 3))

im = plt.imshow(
    heatmap,
    cmap="gray_r",   # 🔥 关键：反转灰度（0=白，32=黑）
    vmin=0,
    vmax=init_r,
    aspect="auto"
)

plt.colorbar(im, label="Rank r")

plt.xticks(
    ticks=range(num_layers),
    labels=[f"L{i}" for i in range(num_layers)],
    rotation=90
)
plt.yticks(
    ticks=range(len(modules)),
    labels=modules
)

plt.xlabel("Transformer Layer Index")
plt.ylabel("Attention Projection Module")
plt.title("Layer-wise Rank Allocation of Preference-Sensitive AdaLoRA (w/o RMB) (λ = 0)")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.close()

print(f"[OK] Heatmap saved to: {OUT_PATH}")
