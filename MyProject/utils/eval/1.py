import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import numpy as np

# ======================
# 路径配置
# ======================
BASE_MODEL_PATH = "/home/data/students_user/2024/lr/model/Llama-3.1-8B-Instruct"
ADAPTER_PATH = "/home/data/students_user/2024/lr/code/MyProject/output/14/adapter"
EVAL_DATA_PATH = "/home/data/students_user/2024/lr/code/MyProject/data/electric_eval_data.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# tokenizer
# ======================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"   # ★ DPO / LLaMA 必须 left padding

# ======================
# 加载 base model + AdaLoRA adapter
# ======================
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# ======================
# 加载评估数据
# electric_eval_data.json:
# { "prompt": ..., "chosen": ..., "rejected": ... }
# ======================
with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

# ======================
# 计算 log-prob score
# ======================
def compute_logprob(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding="max_length"
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    # shift
    logits = logits[:, :-1]
    labels = inputs["input_ids"][:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    token_logp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    # mask padding
    attention_mask = inputs["attention_mask"][:, 1:]
    token_logp = token_logp * attention_mask

    return token_logp.sum(dim=-1).item()

# ======================
# Eval loop
# ======================
accuracies = []
margins = []

for example in tqdm(eval_data, desc="Evaluating DPO Adapter"):
    prompt = example["prompt"]
    chosen = example["chosen"]
    rejected = example["rejected"]

    chosen_text = prompt + chosen
    rejected_text = prompt + rejected

    score_chosen = compute_logprob(chosen_text)
    score_rejected = compute_logprob(rejected_text)

    margin = score_chosen - score_rejected
    acc = 1.0 if margin > 0 else 0.0

    margins.append(margin)
    accuracies.append(acc)

# ======================
# 统计结果
# ======================
print("\n====== DPO Adapter Eval Results ======")
print(f"Accuracy        : {np.mean(accuracies):.4f}")
print(f"Reward Margin   : {np.mean(margins):.4f}")
print(f"Margin Std      : {np.std(margins):.4f}")

"""
====== 2 ======
Accuracy        : 0.6313
Reward Margin   : 29.4013
Margin Std      : 83.9954

====== 5 ======
Accuracy        : 0.6300
Reward Margin   : 29.1333
Margin Std      : 83.8102

====== 14 ======
Accuracy        : 0.6287
Reward Margin   : 28.8267
Margin Std      : 83.8357

"""

