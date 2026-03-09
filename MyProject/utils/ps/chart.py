import json
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# =========================
# 路径配置
# =========================
CONFIG_PATH = "/home/data/students_user/2024/lr/code/MyProject/output/20/adapter/adapter_config.json"

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "20.png"


# =========================
# 读取 JSON
# =========================
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

sensitivity_cache = config["_dpo_sensitivity_cache"]

# =========================
# 按 layer 收集 q / v
# =========================
num_layers = 32
q_values = [0.0] * num_layers
v_values = [0.0] * num_layers

pattern = re.compile(r"model\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)")

for name, value in sensitivity_cache.items():
    match = pattern.match(name)
    if match:
        layer_idx = int(match.group(1))
        proj_type = match.group(2)
        if proj_type == "q_proj":
            q_values[layer_idx] = value
        else:
            v_values[layer_idx] = value

layers = np.arange(num_layers)

# =========================
# 绘图
# =========================
plt.figure(figsize=(10, 5))

plt.plot(
    layers,
    q_values,
    linestyle="-",
    label="q_proj Sensitivity"
)

plt.plot(
    layers,
    v_values,
    linestyle="--",
    label="v_proj Sensitivity"
)

plt.xlabel("Layer Index")
plt.ylabel("Sensitivity Value")
plt.title("DPO Sensitivity Distribution Across Layers")
plt.legend()
plt.grid(True)

plt.tight_layout()

# =========================
# 保存图片
# =========================
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.show()

print(f"Figure saved to: {OUTPUT_PATH}")