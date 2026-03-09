import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
import torch

# 加载基础模型
model_path = "/home/data/students_user/2024/lr/model/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True, 
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载数据集
with open("./data/casual-conversation-poo.json", "r", encoding="utf-8") as f:
    dataset = Dataset.from_list(json.load(f))

from trl.trainer.utils import DPODataCollatorWithPadding

def preprocess_dpo(example):
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"]
    }

dataset = dataset.map(preprocess_dpo)

# DPO配置
training_args = DPOConfig(
    output_dir="./output/dpo",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    max_length=1024,
    max_prompt_length=512,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=5e-5,
    remove_unused_columns=False,
    optim="paged_adamw_8bit"
)

# 初始化DPO训练器
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    # data_collator=DPODataCollatorWithPadding(tokenizer=tokenizer),
)

# 开始训练
trainer.train()

# 保存完整模型
model.save_pretrained("./output/dpo", safe_serialization=True)
tokenizer.save_pretrained("./output/dpo")