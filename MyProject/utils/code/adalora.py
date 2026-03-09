import json
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import AdaLoraConfig, get_peft_model
import torch

# 加载基础模型
model_path = "/home/data/students_user/2024/lr/model/Llama-3.2-1B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# 应用AdaLoRA
ada_lora_config = AdaLoraConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    init_r=12,
    tinit=200,
    tfinal=1000,
    deltaT=10,
    beta1=0.85,
    beta2=0.85,
    orth_reg_weight=0.5,
    total_step=2250,
)
model = get_peft_model(model, ada_lora_config)
model.print_trainable_parameters()

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载并预处理数据集
with open("./data/casual-conversation-poo.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)
    for sample in raw_data:
        sample["instruction"] = sample.pop("prompt")
        sample["output"] = sample.pop("chosen")
        sample.pop("rejected", None)  # ✅ 删除 rejected 字段，防止 Trainer 报错

dataset = Dataset.from_list(raw_data)

# tokenizer 处理，只用 instruction + output 拼接
def preprocess(example):
    text = example["instruction"] + example["output"]
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=1024,
        padding="max_length",
        return_tensors="pt",
    )
    labels = tokenized["input_ids"].clone()
    labels[tokenized["attention_mask"] == 0] = -100  # 忽略 padding
    
    return {
        "input_ids": tokenized["input_ids"][0],  # 去掉 batch 维度
        "attention_mask": tokenized["attention_mask"][0],
        "labels": labels[0],  # ✅ 必须提供 labels
    }


dataset = dataset.map(preprocess, remove_columns=["instruction", "output"])


# 训练配置
training_args = TrainingArguments(
    output_dir="./output/adalora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    optim="paged_adamw_8bit",
    # fp16=True,
    bf16=True,
    remove_unused_columns=False,

    # label_names=["labels"],
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # 因果 LM 用 mlm=False
)
# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    # processing_class=tokenizer,#preprocessor
    data_collator=data_collator,
    # label_names=["labels"],

)

# 开始训练
trainer.train()

# 保存适配器
model.save_pretrained("./output/adalora/adapter")