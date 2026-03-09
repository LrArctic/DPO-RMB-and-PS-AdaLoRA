import json
import peft

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, get_peft_model  # 改用 Lora
import trl
import peft
from peft import PeftModel
print("trl 来自路径:", trl.__file__, trl.__version__)
print("peft 来自路径:", peft.__file__)

import torch

from transformers import TrainerCallback

OUTPUT_DIR = "./output/Llama-3.2-1B-lora"

# 加载基础模型
model_path = "/home/data/students_user/2024/lr/model/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True, 
    device_map="auto",
    torch_dtype=torch.bfloat16  # 节省显存
)

# ✅ rank_pattern（不同层不同秩）
rank_pattern = {
  "model.layers.0.self_attn.q_proj": 1,
  "model.layers.0.self_attn.v_proj": 13,
  "model.layers.1.self_attn.q_proj": 1,
  "model.layers.1.self_attn.v_proj": 15,
  "model.layers.2.self_attn.q_proj": 4,
  "model.layers.2.self_attn.v_proj": 14,
  "model.layers.3.self_attn.q_proj": 4,
  "model.layers.3.self_attn.v_proj": 14,
  "model.layers.4.self_attn.q_proj": 12,
  "model.layers.4.self_attn.v_proj": 14,
  "model.layers.5.self_attn.q_proj": 7,
  "model.layers.5.self_attn.v_proj": 16,
  "model.layers.6.self_attn.q_proj": 10,
  "model.layers.6.self_attn.v_proj": 13,
  "model.layers.7.self_attn.q_proj": 13,
  "model.layers.7.self_attn.v_proj": 16,
  "model.layers.8.self_attn.q_proj": 3,
  "model.layers.8.self_attn.v_proj": 16,
  "model.layers.9.self_attn.q_proj": 1,
  "model.layers.9.self_attn.v_proj": 15,
  "model.layers.10.self_attn.q_proj": 1,
  "model.layers.10.self_attn.v_proj": 4,
  "model.layers.11.self_attn.q_proj": 1,
  "model.layers.11.self_attn.v_proj": 9,
  "model.layers.12.self_attn.q_proj": 1,
  "model.layers.12.self_attn.v_proj": 14,
  "model.layers.13.self_attn.q_proj": 1,
  "model.layers.13.self_attn.v_proj": 9,
  "model.layers.14.self_attn.q_proj": 1,
  "model.layers.14.self_attn.v_proj": 13,
  "model.layers.15.self_attn.q_proj": 1,
  "model.layers.15.self_attn.v_proj": 8
}

# 🔥 应用 LoRA（固定秩）13622
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
    r=16,  # 默认 init_r（会被 rank_pattern 覆盖）
    lora_alpha=32,
    lora_dropout=0.05,
    rank_pattern=rank_pattern
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 打印可训练参数占比

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载数据集
with open("./data/casual-conversation-poo.json", "r", encoding="utf-8") as f:
    dataset = Dataset.from_list(json.load(f))

# DPO配置
training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,  # 梯度累积
    num_train_epochs=3,
    max_length=1024,
    max_prompt_length=512,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=5e-5,
    remove_unused_columns=False,
    optim="paged_adamw_8bit", # 8bit优化器
    logging_dir="./logs",           # tensorboard 日志目录
    report_to="tensorboard",        # 只用 tensorboard
)

# 初始化 DPOTrainer
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer
)

# 开始训练
trainer.train()

# 保存适配器
model.save_pretrained(f"{OUTPUT_DIR}/adapter")

# 如需合并 LoRA 权重再保存：
# model.merge_and_unload()
# model.base_model.save_pretrained(f"{OUTPUT_DIR}/merged_model")
