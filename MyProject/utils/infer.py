import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ====== 配置路径 ======
base_model_path = "/home/data/students_user/2024/lr/code/MyProject/merged-model"
#adapter_path = "/home/data/students_user/2024/lr/code/MyProject/output/Llama-3.2-1B-16-8/adapter"

# ====== 加载基础模型 ======
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",          # 自动放到 GPU 或 CPU
    torch_dtype=torch.bfloat16  # 如果是 bfloat16 训练的
)

# ====== 加载 LoRA Adapter ======
#model = PeftModel.from_pretrained(model, adapter_path)

# ====== 合并 LoRA 权重到基座模型（可选，但推理更快）======
#model = model.merge_and_unload()

# ====== 加载分词器 ======
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ====== 推理 ======
prompt = "你好，请介绍一下你自己。"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

print("生成结果:", tokenizer.decode(outputs[0], skip_special_tokens=True))
