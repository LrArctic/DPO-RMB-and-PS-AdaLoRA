from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import os

# 你的日志路径（支持多个文件，自动合并）
# log_dir = "/home/data/students_user/2024/lr/code/MyProject/output/3/logs/events.out.tfevents.1765300270.imust-Super-Server.62184.0"   # 3
log_dir = "/home/data/students_user/2024/lr/code/MyProject/output/20/logs/events.out.tfevents.1767178978.imust-Super-Server.2486008.0" #2
#log_dir = "/home/data/students_user/2024/lr/code/MyProject/output/2/logs/events.out.tfevents.1765337620.imust-Super-Server.575300.0" #5

# 正确变量名
event_acc = EventAccumulator(log_dir)
event_acc.Reload()                # ←←←← 改成 event_acc

# 查看所有可用的指标（第一次跑建议先打印看看）
print("可用指标：", event_acc.Tags()['scalars'])

# 关键指标（根据你的 logging 名称修改）
tags = {
    "Loss":           "train/loss",
    "Margin":         "train/rewards/margins",
    "Chosen Reward":  "train/rewards/chosen",
    "Rejected Reward":"train/rewards/rejected",
    "Accuracy":       "train/rewards/accuracies",
}

print("\n" + "="*60)
for name, tag in tags.items():
    if tag not in event_acc.Tags()['scalars']:
        print(f"Warning: {tag} 未找到！")
        continue
        
    values = np.array([x.value for x in event_acc.Scalars(tag)])
    
    # 去掉前10% warmup
    values = values[int(0.1 * len(values)):]
    
    mean = values.mean()
    std  = values.std()
    var  = values.var()
    final_100 = values[-100:].mean()
    
    print(f"\n【{name}】")
    print(f"  Mean           : {mean:.4f}")
    print(f"  Std            : {std:.4f}")
    print(f"  Variance       : {var:.5f}   ")
    print(f"  Final 100 steps : {final_100:.4f}")

print("\n" + "="*60)

