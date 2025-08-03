import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dask.optimization import inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

ibd = pd.read_csv('data/s/data2.csv')
X = ibd.drop(columns=['Disease_Name'])
y = ibd['Disease_Name']

#划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1, random_state=42)

# #取prjeb1220数据集为训练集，R语言包中数据集为测试集
# ibd = pd.read_csv('./data/prieb1220_dataset.csv')
# X_train = ibd.drop(columns=['Disease_Name'])
# y_train = ibd['Disease_Name']
#
# ibd = pd.read_csv('./data/r_dataset.csv')
# X_test = ibd.drop(columns=['Disease_Name'])
# y_test = ibd['Disease_Name']
#
# training_accuracy = []
# test_accuracy = []
# ibd_features = ibd.drop(columns=['Disease_Name']).columns

print(ibd.columns)

#数据集维度，多少数据点，每个数据点有多少特征
print("dimension of IBD data: {}".format(ibd.shape))

#目标值（列名）对应的值和个数
print(ibd.groupby('Disease_Name').size())


sns.countplot(ibd['Disease_Name'],label="Count")
ibd.info()

# 检查使用决策树预测ibd的准确性得分
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("训练集准确性: {:.3f}".format(tree.score(X_train, y_train)))
print("测试集准确性: {:.3f}".format(tree.score(X_test, y_test)))

# 上面检查得：使用决策树分类器的训练集准确性为100%而测试集的准确性则差得多，表明树过度拟合不能很好地归纳新数据，需要对树进行预修剪
# 设置max_depth=3,再次进行操作（限制树的深度可减少过度拟合，也会导致训练集的准确性降低，但会提高测试集的准确性）
# tree = DecisionTreeClassifier(max_depth=3, random_state=0)
# tree.fit(X_train, y_train)
# print("训练集准确性: {:.3f}".format(tree.score(X_train, y_train)))
# print("测试集准确性: {:.3f}".format(tree.score(X_test, y_test)))

#可视化决策树预测ibd的特征重要性
# def plot_feature_importances_ibd(model, ibd_features):
#     n_features = len(model.feature_importances_)  # 获取特征重要性的长度
#     plt.figure(figsize=(8,6))
#     plt.barh(range(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), ibd_features)
#     plt.xlabel("Feature importance")
#     plt.ylabel("Feature")
#     plt.ylim(-1, n_features)
#
# # 假设tree是您的模型，ibd_features是特征名称列表
# plot_feature_importances_ibd(tree, ibd_features)
# plt.show()

# def save_feature_importances_to_csv(model, features, filename):
#     # 创建一个DataFrame，其中包含特征名称和对应的特征重要性
#     feature_importances = pd.DataFrame({
#         'Feature': features,
#         'Importance': model.feature_importances_
#     })
#
#     # 将DataFrame写入CSV文件
#     feature_importances.to_csv(filename, index=False)
#
#
# save_feature_importances_to_csv(tree, ibd_features, './data/s/feature_importances.csv')

