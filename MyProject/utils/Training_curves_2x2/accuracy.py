import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 设置全局学术字体 (SCI 常用无衬线字体)
plt.rcParams['font.family'] = 'Arial' 
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def load_and_smooth(path, label, weight=0.8):
    df = pd.read_csv(path)
    # 指数移动平均平滑
    df['smoothed'] = df['Value'].ewm(alpha=1-weight).mean()
    df['Method'] = label # 用于分类
    return df

# 2. 读取并处理两个数据文件
# 建议将 label 改为您的模型名称，如 'Ours (PA-Align)' 和 'Baseline (DPO)'
df1 = load_and_smooth('2_accuracies.csv', 'Baseline', weight=0.9)
df2 = load_and_smooth('20_accuracies_false.csv', 'DPO-RMB+PS-AdaLoRA', weight=0.9)

# 合并数据
df_all = pd.concat([df1, df2])

# 3. 开始绘图
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

# 定义学术配色 (低饱和度蓝色与橙色)
colors = {"Baseline": "#A0A0A0", "DPO-RMB+PS-AdaLoRA": "#1f77b4"} 

# 绘制原始数据 (高透明度粉尘线)
sns.lineplot(data=df_all, x='Step', y='Value', hue='Method', 
             palette=colors, alpha=0.2, legend=False, ax=ax)

# 绘制平滑后的核心曲线 (加粗)
sns.lineplot(data=df_all, x='Step', y='smoothed', hue='Method', 
             palette=colors, linewidth=2, ax=ax)

# 4. SCI 规范化美化
ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
ax.set_ylabel('Train Accuracy (%)', fontsize=12, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)

# 移除上方和右侧的边框 (更显高级)
sns.despine()

# 设置图例 - 放在右下角或左上角
ax.legend(title=None, frameon=True, loc='lower right', fontsize=10)

# 5. 紧凑布局并保存
plt.tight_layout()
plt.savefig('./accuracy_comparison_false.pdf', bbox_inches='tight') # 导出矢量 PDF
plt.show()