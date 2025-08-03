import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示支持
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 模型名称
models = ['LR', 'SVM', 'XGBoost', 'RF', 'Stacking', 'Voting', '改进Stacking']

# 各模型的性能指标（根据实验结果）
accuracy = [0.8017, 0.8467, 0.8505, 0.8338, 0.8658, 0.8536, 0.8658]
precision = [0.7878, 0.7751, 0.8379, 0.8406, 0.8399, 0.8461, 0.8449]
recall = [0.5778, 0.8027, 0.7072, 0.6618, 0.7572, 0.7095, 0.7504]
f1_score = [0.6557, 0.7834, 0.7626, 0.7297, 0.7933, 0.7665, 0.7922]
auc = [0.8380, 0.8948, 0.9016, 0.8891, 0.9129, 0.9116, 0.9140]

# 设置图形大小
plt.figure(figsize=(16, 8))

# 设置柱状图的位置和宽度
x = np.arange(len(models))  # 模型名称的位置
width = 0.15  # 柱子的宽度

# 绘制 Accuracy 柱状图
plt.bar(x - 2*width, accuracy, width, label='Accuracy', color='skyblue')
# 添加数值标签
for i, v in enumerate(accuracy):
    plt.text(x[i] - 2*width, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

# 绘制 Precision 柱状图
plt.bar(x - width, precision, width, label='Precision', color='lightgreen')
# 添加数值标签
for i, v in enumerate(precision):
    plt.text(x[i] - width, v - 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

# 绘制 Recall 柱状图
plt.bar(x, recall, width, label='Recall', color='salmon')
# 添加数值标签
for i, v in enumerate(recall):
    plt.text(x[i], v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

# 绘制 F1-score 柱状图
plt.bar(x + width, f1_score, width, label='F1-score', color='gold')
# 添加数值标签
for i, v in enumerate(f1_score):
    plt.text(x[i] + width, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

# 绘制 AUC 柱状图
plt.bar(x + 2*width, auc, width, label='AUC', color='purple')
# 添加数值标签
for i, v in enumerate(auc):
    plt.text(x[i] + 2*width, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

# 添加图例，将位置设置为左下角
plt.legend(loc='lower left')

# 添加标题和标签
plt.title('模型性能比较', fontsize=14)
plt.xticks(x, models)
plt.ylabel('性能得分', fontsize=12)
plt.xlabel('模型', fontsize=12)

# 保存图形
plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()