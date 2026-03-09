import json
import peft

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from peft import AdaLoraConfig, get_peft_model  # 新增
import trl
import peft
from peft import PeftModel
print("trl 来自路径:", trl.__file__, trl.__version__)
print("peft 来自路径:", peft.__file__)

import torch

from transformers import TrainerCallback

OUTPUT_DIR = "./output/1/"


# 1. 先定义 RMBConfig 类
from dataclasses import dataclass

@dataclass
class RMBConfig:
    enabled: bool = True          # 是否启用 RMB
    size: int = 10000             # Buffer 容量
    batch_size: int = 32          # 每次抽取历史条目数
    mix_ratio: float = 0.2        # 混合比例，历史 loss 占比
    sample_strategy: str = "uniform"  # 采样策略

class AdaLoRAUpdateCallback(TrainerCallback):
    def __init__(self, model, ada_lora_config):
        self.model = model
        self.deltaT = ada_lora_config.deltaT
        self.tinit = ada_lora_config.tinit
        self.tfinal = ada_lora_config.tfinal
        
    def inspect_once(self):
        target_layer = 5  # 写死第4个 q_proj 层（索引从0开始）
        # print(f"\n[检查] ---- AdaLoRA 动态秩更新后打印第 {target_layer} 层的 A/B/E ----")

        count = 0
        for name, module in self.model.named_modules():
            if "q_proj" in name and hasattr(module, "lora_A"):
                if count == target_layer:
                    # print(f"模块: {name}")
                    A = module.lora_A["default"]
                    B = module.lora_B["default"]
                    E = module.lora_E["default"]
                    # print(f"A 形状: {A.shape}, 范数: {A.norm().item():.4f} B 形状: {B.shape}, 范数: {B.norm().item():.4f} E 形状: {E.shape}, 范数: {E.norm().item():.4f}")
                    return
                count += 1
                

    def on_substep_end(self, args, state, control, **kwargs):
        step = state.global_step

        lora_config = self.model.peft_config[self.model.trainable_adapter_name]
        total_step = lora_config.total_step
        middle_step = total_step - self.tfinal

        # 需要触发 update_and_allocate 的时机
        should_allocate = (
            step == self.tinit or
            (step > self.tinit and (step - self.tinit) % self.deltaT == 0) or
            step == middle_step
        )

        if should_allocate:
            # print(f"\n[AdaLoRA] Step {step}: 触发 update_and_allocate()")
            self.model.update_and_allocate(global_step=step)

            # 打印 A/B/E 的变化
            self.inspect_once()


# 加载基础模型
# model_path = "/home/data/students_user/2024/lr/model/Llama-3.2-1B-Instruct"
model_path = "/home/data/students_user/2024/lr/model/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True, 
    device_map="auto",
    torch_dtype=torch.bfloat16  # 节省显存
)

# 🔥 应用 AdaLoRA
ada_lora_config = AdaLoraConfig(    
    target_modules=["q_proj", "v_proj"],  # 目标模块（Llama结构）
    lora_alpha=32,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    init_r=32,  # 初始秩
    target_r=16, #目标秩
    # rank_pattern=rank_pattern,
    #tinit < total_step - tfinal =中期
    
    # tinit=100, # 开始适配的步骤100 
    # tfinal=1250, # 停止适配的步骤400
    # deltaT=10, # 更新间隔 
    # total_step=1398, # ✅ 关键参数！0%|▋

    tinit = 200,        # 建议 200 步开始动态秩分配
    tfinal = 3000,       # 3000 步停止动态秩分配
    deltaT = 50,         # 每 50 步触发一次 allocate
    total_step = 3750,   # 训练总步数
    

    beta1=0.85,  # 重要性分数EMA系数
    beta2=0.85,  # 重要性分数二阶动量
    orth_reg_weight=0.5,  # 正交正则化强度
    
)
model = get_peft_model(model, ada_lora_config)
model.print_trainable_parameters()  # 打印可训练参数占比

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载数据集

# with open("./data/casual-conversation-poo.json", "r", encoding="utf-8") as f:
#     dataset = Dataset.from_list(json.load(f))
with open("/home/data/students_user/2024/lr/code/MyProject/data/electric_train_data.json", "r", encoding="utf-8") as f:
    dataset = Dataset.from_list(json.load(f))

# Reward Memory Buffer
RMB_config = RMBConfig(
    enabled=True,         # 启用 RMB
    size=10000,           # buffer 容量（推荐 5k-50k）
    batch_size=32,        # 每步从 buffer 抽多少条来 mix
    mix_ratio=0.2,        # 历史 loss 占最终 loss 的比例（0.0 - 1.0）
    sample_strategy='uniform'  # 当前仅支持 uniform
)

# DPO配置
training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,  # 梯度累积
    num_train_epochs=3,
    max_length=2048,
    max_prompt_length=512,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=5e-5,
    remove_unused_columns=False,
    optim="paged_adamw_8bit", # 8bit优化器

    rpo_alpha=0.1,
    # optim="adamw_torch",
    logging_dir="{OUTPUT_DIR}/logs",           # tensorboard 日志目录
    report_to="tensorboard",        # 只用 tensorboard，不用 wandb
    # report_to="none"  # ✅ 禁用 wandb 和所有日志后端
    rmb_enabled=RMB_config.enabled,
    rmb_size=RMB_config.size,
    rmb_batch_size=RMB_config.batch_size,
    rmb_mix_ratio=RMB_config.mix_ratio,
    rmb_strategy=RMB_config.sample_strategy
)

# 初始化DPO训练器
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,  # 正确传法
    # tokenizer=tokenizer,#还是让加这个
    callbacks = [
    AdaLoRAUpdateCallback(model, ada_lora_config)
    ]

)

# 开始训练
trainer.train()

# 保存适配器
model.save_pretrained(f"{OUTPUT_DIR}/adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/adapter")