import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==================== 1. 配置路径 ====================
# BASE_MODEL_PATH = "/home/data/students_user/2024/lr/model/Llama-3.1-8B-Instruct"

BASE_MODEL_PATH = "/home/data/students_user/2024/lr/model/Qwen2.5-7B-Instruct"

DPO_ADAPTER_PATH = "/home/data/students_user/2024/lr/code/MyProject/output/21/adapter"
EVAL_DATA_PATH = "/home/data/students_user/2024/lr/code/MyProject/data/electric_eval_data.json"
SAVE_PATH = "21-2.json"

# ==================== 2. 加载模型与分词器 ====================
print("正在加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("正在加载模型并应用 Adapter...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
# 加载你的 AdaLoRA Adapter
model = PeftModel.from_pretrained(model, DPO_ADAPTER_PATH)
model.eval()

# ==================== 3. 读取前 20 条数据 ====================
with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
    eval_data = json.load(f)[:20]

# ==================== 4. 循环生成 ====================
results = []
print(f"开始生成前 20 条数据的回复...")

for item in tqdm(eval_data):
    prompt = item["prompt"]
    reference = item.get("chosen", "") # 拿 chosen 作为参考答案
    
    # 构造对话格式
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码生成结果
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response_ours = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    results.append({
        "prompt": prompt,
        #"reference_chosen": reference,
        "response_ours": response_ours
    })

# ==================== 5. 保存结果 ====================
with open(SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n生成完毕！结果已保存至: {SAVE_PATH}")