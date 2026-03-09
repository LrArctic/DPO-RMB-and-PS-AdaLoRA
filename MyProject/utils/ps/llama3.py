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
OUTPUT_PATH = SCRIPT_DIR / "llama3_sensitivity.pdf"

# =========================
# 读取 JSON
# =========================
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

sensitivity_cache = config.get("_dpo_sensitivity_cache", {})

# =========================
# 按 layer 收集 q / v (Llama-3.1-8B 标准为 32 层)
# =========================
num_layers = 32 
q_values = [0.0] * num_layers
v_values = [0.0] * num_layers

# 匹配正则
pattern = re.compile(r"model\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)")

for name, value in sensitivity_cache.items():
    match = pattern.match(name)
    if match:
        layer_idx = int(match.group(1))
        if layer_idx < num_layers:
            proj_type = match.group(2)
            if proj_type == "q_proj":
                q_values[layer_idx] = value
            else:
                v_values[layer_idx] = value

# =========================
# 纵坐标换算 (保持与 Qwen 一致的 100 倍缩放)
# =========================
SCALE_FACTOR = 100
q_values_scaled = np.array(q_values) * SCALE_FACTOR
v_values_scaled = np.array(v_values) * SCALE_FACTOR

layers = np.arange(num_layers)

# =========================
# 绘图 (与 Qwen 风格完全一致)
# =========================
plt.figure(figsize=(9, 4.5))

# 使用相同的颜色和标记点
plt.plot(
    layers, 
    q_values_scaled, 
    color='#1f77b4', 
    marker='o', 
    markersize=4,
    linewidth=1.5, 
    label=r"$W_q$ Sensitivity ($\times 10^{-2}$)"
)

plt.plot(
    layers, 
    v_values_scaled, 
    color='#ff7f0e', 
    linestyle="--", 
    marker='s', 
    markersize=4,
    linewidth=1.5, 
    label=r"$W_v$ Sensitivity ($\times 10^{-2}$)"
)

# 坐标轴与标题
plt.xlabel("Layer Index", fontsize=12)
plt.ylabel("Scaled Sensitivity Value", fontsize=12)
plt.title("Preference Sensitivity Distribution of Llama-3.1-8B-Instruct", fontsize=13)

# 设置刻度 (32 层模型建议刻度步长为 4 或 2)
plt.xticks(np.arange(0, num_layers, 2))
plt.xlim(-0.5, num_layers - 0.5)

plt.legend(frameon=True, loc='lower right') # Llama 通常左侧较空
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()

# =========================
# 保存图片
# =========================
plt.savefig(OUTPUT_PATH, bbox_inches="tight")
plt.show()

print(f"Figure saved to: {OUTPUT_PATH}")