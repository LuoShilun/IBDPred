import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置支持中文的字体
plt.rcParams['font.family'] = 'SimHei'  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载CSV文件
file_path = './data/samples_health.csv'  # 如果需要，请更新为正确的文件路径
df = pd.read_csv(file_path)
# 查看数据的前几行，理解数据结构
print(df.head())
# 统计健康状态的分布
health_status_counts = df['Health State'].value_counts()
# 创建疾病名称的类别列表
disease_categories = ['Healthy', 'Obesity', 'UC', 'IBD', 'CD']

# 统计不同疾病类型下的样本数量
disease_counts = {category: 0 for category in disease_categories}
# 统计每种健康状态下的疾病类型
for health_state in ['Healthy', 'Diseased']:
    if health_state == 'Healthy':
        disease_counts['Healthy'] = health_status_counts.get('Healthy', 0)
    else:
        # 仅对Diseased状态进行疾病名称的计数
        diseased_samples = df[df['Health State'] == 'Diseased']
        for disease in disease_categories[1:]:  # 跳过'Healthy'
            disease_counts[disease] = len(diseased_samples[diseased_samples['Disease Name'] == disease])

# 显示统计结果
print("\n健康状态和疾病名称的统计分布:")
print(disease_counts)

# 绘制并排条形图
labels = ['Healthy', 'Obesity', 'UC', 'IBD', 'CD']
values = [disease_counts['Healthy'], disease_counts['Obesity'], disease_counts['UC'], disease_counts['IBD'], disease_counts['CD']]

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, values, color=['#4CAF50', '#FF9800', '#FF5722', '#FFC107', '#9C27B0'], edgecolor='black')

# 设置标题和标签
plt.title('不同健康状态下的疾病分布')
plt.xlabel('健康状态及疾病')
plt.ylabel('样本数量')

# 在每个条形图上标注具体的样本数量
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha='center', va='bottom', fontsize=10)

# 显示图形
plt.tight_layout()
plt.show()
