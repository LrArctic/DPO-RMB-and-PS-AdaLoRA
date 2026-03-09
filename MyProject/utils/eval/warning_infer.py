import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import bert_score

# ==================== 配置 ====================
BASE_MODEL = "/home/data/students_user/2024/lr/model/Llama-3.1-8B-Instruct"
DPO_ADAPTER = "/home/data/students_user/2024/lr/code/MyProject/output/2/adapter"  # 你的 DPO 模型
EVAL_DATA = "/home/data/students_user/2024/lr/code/MyProject/data/electric_eval_data.json"
OUTPUT_JSON = "dpo_eval_results_domain_specific.json"

# ==================== 加载 Tokenizer 和模型 ====================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading DPO-aligned model for generation...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, DPO_ADAPTER)
# 可选：合并加速生成（推荐！）
model = model.merge_and_unload()
model.eval()

# ==================== 加载评估数据集 ====================
with open(EVAL_DATA, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

eval_dataset = Dataset.from_list(raw_data)  # 要求每条有 "prompt", "chosen"（参考答案）

# ==================== 初始化自动指标计算器 ====================
smoother = SmoothingFunction()
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# ==================== 生成与评估 ====================
results = []
bleu_scores = []
rouge_l_scores = []
generation_lengths = []

print("Generating responses and computing metrics...")
for item in tqdm(eval_dataset, desc="Evaluating"):
    prompt = item["prompt"]
    
    # 生成
    inputs = tokenizer(prompt + "请使用英文回答。", return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(generated_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    reference = item["chosen"].strip()
    
    # 计算自动指标
    ref_tokens = reference.split()
    gen_tokens = generated_text.split()
    
    # BLEU-4
    bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoother.method1)
    bleu_scores.append(bleu)
    
    # ROUGE-L
    rouge = scorer.score(reference, generated_text)['rougeL'].fmeasure
    rouge_l_scores.append(rouge)
    
    # 长度
    gen_len = len(gen_tokens)
    generation_lengths.append(gen_len)
    
    # 保存结果供人工评估
    results.append({
        "prompt": prompt,
        "reference (chosen)": reference,
        "generated": generated_text,
        "bleu": bleu,
        "rouge_l": rouge,
        "length": gen_len
    })

# ==================== BERTScore（批量计算，更快） ====================
print("Computing BERTScore...")
candidates = [r["generated"] for r in results]
references = [r["reference (chosen)"] for r in results]
P, R, F = bert_score.score(candidates, references, lang="en", verbose=False)
bert_f = F.mean().item()

# ==================== 输出结果 ====================
print("\n=== Domain-Specific Evaluation Results ===")
print(f"BLEU-4          : {np.mean(bleu_scores):.4f}")
print(f"ROUGE-L         : {np.mean(rouge_l_scores):.4f}")
print(f"BERTScore (F1)  : {bert_f:.4f}")
print(f"Avg Length      : {np.mean(generation_lengths):.1f} tokens")
print(f"Std Length      : {np.std(generation_lengths):.1f}")

# ==================== 保存详细结果（用于人工评估） ====================
df = pd.DataFrame(results)
df.to_json(OUTPUT_JSON, orient="records", indent=2, force_ascii=False)
print(f"\n详细结果已保存到 {OUTPUT_JSON}，可抽样进行人工评估")

# 可选：保存统计指标
metrics = {
    "bleu": np.mean(bleu_scores),
    "rouge_l": np.mean(rouge_l_scores),
    "bertscore_f1": bert_f,
    "avg_length": np.mean(generation_lengths),
    "std_length": np.std(generation_lengths),
    "num_samples": len(results)
}
with open("dpo_eval_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

"""
2

"""