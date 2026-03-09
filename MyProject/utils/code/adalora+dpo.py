# 222_fixed.py
# 修改点：
# - 提取 rank pattern 的鲁棒逻辑
# - 确保保存 adapter（adapter_config.json + adapter_model.bin）
# - Stage2 安全加载 adapter 并冻结 base（可选）
# - 若本地 peft/adalora 有不同 API，会打印友好提示

import os
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import get_peft_model, AdaLoraConfig, PeftModel
from trl import DPOTrainer, DPOConfig

OUTPUT_DIR = "./output/Llama-3.2-1B-adalora+dpo"
MID_SAVE = f"{OUTPUT_DIR}/mid_ablation"        # 中期 adapter 保存目录（必须包含 adapter_config.json + adapter_model.bin）
FINAL_ADAPTER = f"{OUTPUT_DIR}/adapter_final" # 最终 adapter 保存目录
BASE_PATH = "/home/data/students_user/2024/lr/model/Llama-3.2-1B-Instruct"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MID_SAVE, exist_ok=True)
os.makedirs(FINAL_ADAPTER, exist_ok=True)

# ---------- 辅助函数 ----------
def save_rank_pattern(rank_pattern, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rank_pattern, f, indent=2, ensure_ascii=False)
    print(f"[INFO] rank_pattern saved -> {path}")

def load_rank_pattern(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_rank_pattern(model):
    """
    尝试多种方式从 model 中提取 rank pattern（兼容不同 peft/adalora 实现）。
    返回 rank_pattern（dict）或抛出 ValueError。
    """
    # 1) 直接方法（如果实现了）
    try:
        rp = model.get_rank_pattern()
        print("[DEBUG] rank_pattern from model.get_rank_pattern()")
        return rp
    except Exception:
        pass

    # 2) 在 base_model / inner model 上查找
    candidates = []
    # unwrap PeftModel -> .base_model or .model
    try:
        base = getattr(model, "base_model", None) or getattr(model, "model", None)
        if base is not None:
            candidates.append(base)
    except Exception:
        pass

    # 3) 在对象树中查找 rankallocator
    for obj in [model] + candidates:
        if obj is None:
            continue
        # 常见命名：rankallocator, rank_allocator, rank_alloc
        for attrname in ("rankallocator", "rank_allocator", "rank_alloc", "rankAllocator"):
            ra = getattr(obj, attrname, None)
            if ra is not None:
                # 尝试常用方法
                for meth in ("get_rank_pattern", "export_rank_pattern", "rank_pattern", "get_pattern"):
                    try:
                        fn = getattr(ra, meth)
                        rp = fn() if callable(fn) else fn
                        print(f"[DEBUG] rank_pattern from {obj.__class__.__name__}.{attrname}.{meth}")
                        return rp
                    except Exception:
                        # 继续尝试其他方法
                        pass

    # 4) 如果 ada-lora 层以某种结构保存 r 信息，尝试遍历 named_modules
    # 尝试查找名为 ada 或 adalora 的子模块里可能保存的 rank
    try:
        for name, m in model.named_modules():
            if "adalora" in name.lower() or "ada_lora" in name.lower() or "adalora" in type(m).__name__.lower():
                # 常见属性：rank_pattern, rank_dict, r_list
                for prop in ("rank_pattern", "rank_dict", "ranks", "r"):
                    if hasattr(m, prop):
                        rp = getattr(m, prop)
                        print(f"[DEBUG] rank_pattern from module {name}.{prop}")
                        return rp
    except Exception:
        pass

    raise ValueError("无法从 model 中提取 rank_pattern 。请检查你的 AdaLoRA 实现是否提供 get_rank_pattern 或在 rankallocator 中保存了 rank 信息。")

def apply_rank_pattern_to_model(peft_model, rank_pattern):
    """
    尝试应用 rank pattern 到 peft_model 上（不同实现 API 不同，做多种尝试）。
    如果没有可用 API，会打印提示并返回 False。
    """
    # 常见 API: peft_model.apply_rank_pattern / peft_model.base_model.apply_rank_pattern
    try:
        if hasattr(peft_model, "apply_rank_pattern"):
            peft_model.apply_rank_pattern(rank_pattern)
            print("[INFO] Applied rank pattern via peft_model.apply_rank_pattern")
            return True
    except Exception:
        pass

    try:
        base = getattr(peft_model, "base_model", None)
        if base is not None and hasattr(base, "apply_rank_pattern"):
            base.apply_rank_pattern(rank_pattern)
            print("[INFO] Applied rank pattern via base_model.apply_rank_pattern")
            return True
    except Exception:
        pass

    # 尝试在 rankallocator 上调用
    try:
        for name, m in peft_model.named_modules():
            if "rankallocator" in name.lower() or hasattr(m, "update_and_allocate"):
                ra = m
                if hasattr(ra, "apply_rank_pattern"):
                    ra.apply_rank_pattern(rank_pattern)
                    print(f"[INFO] Applied rank pattern via module {name}.apply_rank_pattern")
                    return True
    except Exception:
        pass

    print("[WARN] 未找到自动应用 rank_pattern 的 API。请根据你的 AdaLoRA 实现手动应用 rank_pattern（或把函数名告诉我以便我帮你改脚本）。")
    return False

def freeze_base_model_params(peft_model, whitelist_adapter_keywords=("lora", "adapter", "adalora", "peft")):
    """
    冻结基础模型参数（使 adapter/LORA 参数保持可训练）。
    若你希望只训练 LoRA/A/B/E，请设置 FREEZE_BASE = True。
    """
    count_total = 0
    count_frozen = 0
    for n, p in peft_model.named_parameters():
        count_total += 1
        # 若名字包含 whitelist 中的任意关键词，我们认为这是 adapter/LoRA 参数 -> keep trainable
        if any(k in n.lower() for k in whitelist_adapter_keywords):
            p.requires_grad = True
        else:
            p.requires_grad = False
            count_frozen += 1
    print(f"[INFO] freeze_base_model_params: frozen {count_frozen}/{count_total} parameters (base frozen, adapters kept trainable).")

# ---------- Callback（和之前类似，但更稳健） ----------
class AdaLoRAUpdateCallback(TrainerCallback):
    def __init__(self, model, ada_lora_config):
        self.model = model
        self.deltaT = ada_lora_config.deltaT
        self.tinit = ada_lora_config.tinit
        self.tfinal = ada_lora_config.tfinal

    def on_substep_end(self, args, state, control, **kwargs):
        # TRL/DPO 的回调点，如果可用，优先这里（通常在 backward 后）
        self._maybe_update(state, control)

    def on_step_end(self, args, state, control, **kwargs):
        # 兼容：先检查是否至少有一个参数含 grad
        any_grad = False
        for n, p in self.model.named_parameters():
            if p.grad is not None:
                any_grad = True
                break
        if not any_grad:
            return control
        self._maybe_update(state, control)

    def _maybe_update(self, state, control):
        step = state.global_step
        # 从 peft_config 读取 total_step（尽量防守）
        try:
            lora_config = self.model.peft_config[self.model.trainable_adapter_name]
            total_step = lora_config.total_step
        except Exception:
            total_step = getattr(self.model, "total_step", None) or 0
        middle_step = total_step - self.tfinal if total_step else None

        if step == self.tinit or (step > self.tinit and (step - self.tinit) % self.deltaT == 0) or (middle_step and step == middle_step):
            if hasattr(self.model, "update_and_allocate"):
                print(f"[DEBUG] Step {step}: 触发 update_and_allocate()")
                try:
                    self.model.update_and_allocate(global_step=step)
                except TypeError:
                    # 有些实现 update_and_allocate 不接受参数
                    self.model.update_and_allocate()

# =====================
# 1) Load base model and attach AdaLoRA
# =====================
model = AutoModelForCausalLM.from_pretrained(
    BASE_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

ada_cfg = AdaLoraConfig(
    init_r=32,
    target_r=16,
    target_modules=["q_proj", "v_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    tinit=100,
    tfinal=1250,
    total_step=1398,
    deltaT=10,
    beta1=0.85,
    beta2=0.85,
    orth_reg_weight=0.5,
)
model = get_peft_model(model, ada_cfg)

tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load DPO dataset
with open("./data/casual-conversation-poo.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)
dataset = Dataset.from_list(raw_data)

# DPO args
args = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    max_length=1024,
    max_prompt_length=512,
    logging_steps=10,
    save_strategy="no",
    learning_rate=5e-5,
    remove_unused_columns=False,
    # optim="paged_adamw_8bit",
    optim="adamw_torch",  
    logging_dir="./logs",
    report_to="tensorboard",
)

# ========== Stage 1: AdaLoRA 动态收集重要性 ==========
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=args,
    train_dataset=dataset,
    processing_class=tokenizer,
    callbacks=[AdaLoRAUpdateCallback(model, ada_cfg)],
)

print("===== Stage 1: AdaLoRA importance collecting & mid pruning =====")
trainer.train()

# 尝试提取 rank_pattern（多路径）
try:
    rank_pattern = extract_rank_pattern(model)
    save_rank_pattern(rank_pattern, f"{MID_SAVE}/rank_pattern.json")
except Exception as e:
    print("[WARN] 无法自动提取 rank_pattern:", str(e))
    print("[WARN] 如果你已经单独保存了 rank_pattern，请把它放到", f"{MID_SAVE}/rank_pattern.json")
    rank_pattern = None

# *关键*：保存 adapter（adapter_config.json + adapter_model.bin）
# 这一步非常重要，否则 PeftModel.from_pretrained 无法加载
print("[INFO] 保存中期 adapter 到", MID_SAVE)
model.save_pretrained(MID_SAVE)
print("[INFO] Done saving adapter. 目录内容：", os.listdir(MID_SAVE))

# ========== Stage 2: 使用冻结后的 A/B/E 继续训练（加载 adapter） ==========
print("===== Stage 2: Restart with frozen LoRA matrices =====")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# 加载保存好的 LoRA adapter（目录需包含 adapter_config.json + adapter_model.bin）
try:
    model2 = PeftModel.from_pretrained(base_model, MID_SAVE)
    print("[INFO] Successfully loaded adapter into base model.")
except Exception as e:
    print("[ERROR] PeftModel.from_pretrained failed:", str(e))
    raise

# 若 Stage1 未能提取到 rank_pattern，但保存了 rank 文件，可尝试加载
if rank_pattern is None and os.path.exists(f"{MID_SAVE}/rank_pattern.json"):
    rank_pattern = load_rank_pattern(f"{MID_SAVE}/rank_pattern.json")

# 应用 rank_pattern（如果可用）
if rank_pattern is not None:
    applied = apply_rank_pattern_to_model(model2, rank_pattern)
    if not applied:
        print("[WARN] 未能自动 apply rank_pattern，请手动根据你的实现应用。")

# 冻结 base model（只训练 adapter/LoRA 参数）——通常训练 adapter 时这样做
FREEZE_BASE = True
if FREEZE_BASE:
    freeze_base_model_params(model2)
else:
    print("[INFO] 不冻结 base parameters（FREEZE_BASE=False）")

# 继续 DPO 训练（第二阶段）
trainer2 = DPOTrainer(
    model=model2,
    ref_model=None,
    args=args,
    train_dataset=dataset,
    processing_class=tokenizer,
)
print("===== Stage 2: Continue DPO training (adapter parameters trainable) =====")
trainer2.train()

# 保存最终 adapter
model2.save_pretrained(FINAL_ADAPTER)
print(f"[INFO] Training completed. Final adapter saved to {FINAL_ADAPTER}")
