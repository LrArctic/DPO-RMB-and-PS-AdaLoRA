import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "/home/data/students_user/2024/lr/model/Llama-3.1-8B-Instruct"
ADAPTER_PATH = "/home/data/students_user/2024/lr/code/MyProject/output/Llama-3.1-8B-Instruct/adapter"
OUTPUT = "./merged-model"

print(">>> Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

print(">>> Loading adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print(">>> Merging LoRA/AdaLoRA weights...")
model = model.merge_and_unload()   # ★ 关键步骤：合并 LoRA 到基座

print(">>> Saving merged full model...")
model.save_pretrained(OUTPUT)

# 可选：保存 tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(OUTPUT)

print(">>> Done! Merged model saved to:", OUTPUT)
