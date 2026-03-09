# 文件名: reward_score_ours.py

import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import numpy as np

# ==================== 配置路径 ====================
REWARD_MODEL_PATH = "/home/data/students_user/2024/lr/model/Llama-3.1-8B-Instruct_Reward_model"
EVAL_DATA_PATH = "/home/data/students_user/2024/lr/code/MyProject/utils/gpt4_eval/21-2.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== 加载 tokenizer 和 reward model ====================
print(f"Loading reward model from {REWARD_MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_PATH)

# 关键：use_fast=False 有时能避免某些 tokenizer 问题，尤其是自定义模型
# 如果 pad_token 不存在，手动设置
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    REWARD_MODEL_PATH,
    torch_dtype=torch.bfloat16,  # 推荐使用 bfloat16 节省显存并保持精度
    device_map="auto",           # 自动分配到可用 GPU
    # attn_implementation="flash_attention_2"  # 如果你安装了 flash-attn，可加速
)

model.eval()
print(f"Model loaded on {DEVICE}")

# ==================== 加载评估数据 ====================
with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples from {EVAL_DATA_PATH}")

# 确保数据格式正确：每条有 "prompt", "reference_chosen", "response_ours"
assert all("response_ours" in item for item in data), "数据中缺少 response_ours 字段！"

# ==================== 构造输入（Reward Model 标准输入格式）===================
# 标准做法：将 prompt + response 拼接，作为一个序列输入
# 很多 Reward Model 在训练时就是这样处理的

def build_input_text(prompt, response):
    # 根据 Llama-3 Instruct 风格，常用格式：
    # <bos>prompt\n\nResponse: response<eos>
    # 或者直接拼接 prompt + response
    # 建议和训练时保持一致！如果你训练时用了特殊格式，这里也要一样
    return f"{prompt.strip()}\n\nResponse: {response.strip()}"

scores = []

print("Computing reward scores for response_ours...")
with torch.no_grad():
    for item in tqdm(data, desc="Scoring"):
        prompt = item["prompt"]
        response = item["response_ours"]

        input_text = build_input_text(prompt, response)

        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=8192,        # Llama-3.1 支持长上下文
            padding=False
        ).to(model.device)

        outputs = model(**inputs)
        # Reward Model 的输出是 logits，第一个维度通常是分数（正值越高越好）
        score = outputs.logits[0, 0].item()  # 取 [batch=1, num_labels=1] 中的值

        scores.append(score)

# ==================== 计算统计结果 ====================
avg_score = np.mean(scores)
std_score = np.std(scores)
max_score = np.max(scores)
min_score = np.min(scores)

print("\n=== Reward Model Scoring Results (on response_ours) ===")
print(f"Number of samples     : {len(scores)}")
print(f"Average Reward Score  : {avg_score:.4f}")
print(f"Std Deviation         : {std_score:.4f}")
print(f"Max Score             : {max_score:.4f}")
print(f"Min Score             : {min_score:.4f}")

# ==================== 可选：保存详细分数到文件 ====================
results = []
for i, item in enumerate(data):
    results.append({
        "index": i,
        "prompt": item["prompt"],
        "response_ours": item["response_ours"],
        "reward_score": scores[i]
    })

output_detail_path = "reward_scores_21-2.json"
with open(output_detail_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n详细打分结果已保存到: {output_detail_path}")
print("Done!")

"""
2
Number of samples     : 20
Average Reward Score  : 0.7732
Std Deviation         : 1.3077
Max Score             : 2.5156
Min Score             : -1.6172
20
Number of samples     : 20
Average Reward Score  : 0.8870
Std Deviation         : 1.0695
Max Score             : 2.3125
Min Score             : -2.1250

21
Number of samples     : 20
Average Reward Score  : 0.2774
Std Deviation         : 1.2323
Max Score             : 1.9688
Min Score             : -1.8203

22
Number of samples     : 20
Average Reward Score  : 0.2858
Std Deviation         : 1.1519
Max Score             : 2.6875
Min Score             : -2.1875
"""