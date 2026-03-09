import torch
import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import DPOConfig, DPOTrainer

# 1. 路径配置2/3 5 14
# BASE_MODEL_PATH = "/home/data/students_user/2024/lr/model/Llama-3.1-8B-Instruct"

BASE_MODEL_PATH = "/home/data/students_user/2024/lr/model/Qwen2.5-7B-Instruct"

DPO_ADAPTER_PATH = "/home/data/students_user/2024/lr/code/MyProject/output/21/adapter"
EVAL_DATA_PATH = "/home/data/students_user/2024/lr/code/MyProject/data/electric_eval_data.json"

# 2. 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. 加载基础模型与 AdaLoRA Adapter
# 注意：虽然训练时用了 AdaLoraConfig，但评估加载时 PeftModel 会自动从 adapter 目录读取配置
print("正在加载基础模型...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print("正在加载 AdaLoRA Adapter...")
# 加载 adapter，is_trainable=False 确保不会意外触发参数更新
model = PeftModel.from_pretrained(model, DPO_ADAPTER_PATH, is_trainable=False)
model.eval()

# 4. 加载评估数据
with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
    eval_data = json.load(f)
eval_dataset = Dataset.from_list(eval_data)

# 5. DPO 评估配置
# 关键：我们不运行训练，只需配置基础的 DPO 参数
# 5. DPO 评估配置 - 将参数写在这里
eval_args = DPOConfig(
    output_dir="./eval_tmp",
    per_device_eval_batch_size=4, 
    max_length=2048,              # 移到这里
    max_prompt_length=512,        # 移到这里
    beta=0.1,                     # 移到这里
    bf16=True,
    remove_unused_columns=False,
    report_to="none"
)
#######################
# 加载参考模型（必须是训练前的原始 Instruct 模型）
ref_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
ref_model.eval()
############
# 6. 初始化 Trainer - 移除重复或不支持的参数
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model, ######################
    args=eval_args,               # Config 会自动把 beta 等传给 Trainer
    train_dataset=eval_dataset,   # 避开 NoneType 报错
    eval_dataset=eval_dataset,
    processing_class=tokenizer,   # 新版本建议使用 processing_class 替代 tokenizer
)

# 7. 执行评估
print("\n--- 开始在测试集上计算 DPO 指标 ---")
# 直接调用 evaluate()，它不会触发训练
metrics = trainer.evaluate()

# 8. 输出结果
print("\n" + "="*30)
print("评估指标结果:")
for key, value in metrics.items():
    if "eval_" in key:
        print(f"{key:25}: {value}")
print("="*30)


"""
1
评估指标结果:
eval_loss                : 0.11411638557910919
eval_model_preparation_time: 0.0078
eval_runtime             : 715.6446
eval_samples_per_second  : 2.096
eval_steps_per_second    : 0.524
eval_rewards/chosen      : 3.4074127674102783
eval_rewards/rejected    : -2.675046443939209
eval_rewards/accuracies  : 0.9553333520889282
eval_rewards/margins     : 6.082459449768066
eval_logps/chosen        : -302.36993408203125
eval_logps/rejected      : -381.8675842285156
eval_logits/chosen       : -1.5541969537734985
eval_logits/rejected     : -1.4028582572937012
2
==============================
评估指标结果:
eval_loss                : 0.3239080011844635
eval_model_preparation_time: 0.0084
eval_runtime             : 417.3591
eval_samples_per_second  : 3.594
eval_steps_per_second    : 0.899
eval_rewards/chosen      : 0.2569979727268219
eval_rewards/rejected    : -2.2423508167266846
eval_rewards/accuracies  : 0.8606666922569275
eval_rewards/margins     : 2.4993488788604736
eval_logps/chosen        : -333.87408447265625
eval_logps/rejected      : -377.5406188964844
eval_logits/chosen       : -0.26824384927749634
eval_logits/rejected     : -0.13530495762825012
==============================
3
==============================
评估指标结果:
eval_loss                : 0.31787389516830444
eval_model_preparation_time: 0.0084
eval_runtime             : 726.5979
eval_samples_per_second  : 2.064
eval_steps_per_second    : 0.516
eval_rewards/chosen      : 0.2531915605068207
eval_rewards/rejected    : -2.2786030769348145
eval_rewards/accuracies  : 0.8579999804496765
eval_rewards/margins     : 2.531794548034668
eval_logps/chosen        : -333.91217041015625
eval_logps/rejected      : -377.9031677246094
eval_logits/chosen       : -0.2717534303665161
eval_logits/rejected     : -0.133947491645813
5
==============================
评估指标结果:
eval_loss                : 0.3147447109222412
eval_model_preparation_time: 0.0084
eval_runtime             : 419.9458
eval_samples_per_second  : 3.572
eval_steps_per_second    : 0.893
eval_rewards/chosen      : 0.2895682752132416
eval_rewards/rejected    : -2.2314090728759766
eval_rewards/accuracies  : 0.8646666407585144
eval_rewards/margins     : 2.520977020263672
eval_logps/chosen        : -333.54840087890625
eval_logps/rejected      : -377.43121337890625
eval_logits/chosen       : -0.2826521396636963
eval_logits/rejected     : -0.14611414074897766
==============================
14
评估指标结果:
eval_loss                : 0.31784024834632874
eval_model_preparation_time: 0.0084
eval_runtime             : 416.0053
eval_samples_per_second  : 3.606
eval_steps_per_second    : 0.901
eval_rewards/chosen      : 0.26562413573265076
eval_rewards/rejected    : -2.210465431213379
eval_rewards/accuracies  : 0.8619999885559082
eval_rewards/margins     : 2.4760892391204834
eval_logps/chosen        : -333.787841796875
eval_logps/rejected      : -377.2218017578125
eval_logits/chosen       : -0.2731236517429352
eval_logits/rejected     : -0.14000988006591797

19
评估指标结果:
eval_loss                : 0.3293127715587616
eval_model_preparation_time: 0.0086
eval_runtime             : 722.9887
eval_samples_per_second  : 2.075
eval_steps_per_second    : 0.519
eval_rewards/chosen      : 0.032379355281591415
eval_rewards/rejected    : -2.2618699073791504
eval_rewards/accuracies  : 0.8533333539962769
eval_rewards/margins     : 2.2942492961883545
eval_logps/chosen        : -336.12030029296875
eval_logps/rejected      : -377.73583984375
eval_logits/chosen       : -0.35360416769981384
eval_logits/rejected     : -0.2207469493150711

20
评估指标结果:
eval_loss                : 0.32847675681114197
eval_runtime             : 407.3025
eval_samples_per_second  : 3.683
eval_steps_per_second    : 0.921
eval_rewards/chosen      : 0.2474740594625473
eval_rewards/rejected    : -2.1769859790802
eval_rewards/accuracies  : 0.8566666841506958
eval_rewards/margins     : 2.4244601726531982
eval_logps/chosen        : -333.9546813964844
eval_logps/rejected      : -376.9253234863281
eval_logits/chosen       : -0.2858944535255432
eval_logits/rejected     : -0.15621212124824524
21
eval_loss                : 0.3302592933177948
eval_runtime             : 392.9573
eval_samples_per_second  : 3.817
eval_steps_per_second    : 0.954
eval_rewards/chosen      : 1.3847501277923584
eval_rewards/rejected    : -0.31935468316078186
eval_rewards/accuracies  : 0.85 999723434448
eval_rewards/margins     : 1.7041047811508179
eval_logps/chosen        : -293.621337890625
eval_logps/rejected      : -319.7706604003906
eval_logits/chosen       : -0.06688765436410904
eval_logits/rejected     : 0.004487468861043453
22
eval_loss                : 0.32546642422676086
eval_runtime             : 398.2525
eval_samples_per_second  : 3.766
eval_steps_per_second    : 0.942
eval_rewards/chosen      : 1.3649336099624634
eval_rewards/rejected    : -0.3702046573162079
eval_rewards/accuracies  : 0.8526666760444641
eval_rewards/margins     : 1.7351382970809937
eval_logps/chosen        : -293.90399169921875
eval_logps/rejected      : -320.28265380859375
eval_logits/chosen       : -0.09285137802362442
eval_logits/rejected     : -0.01038513146340847
"""