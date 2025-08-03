import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dask.optimization import inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

plt.rcParams['font.sans-serif'] = ['SimHei']

# ibd = pd.read_csv('data/s/data2.csv')
# X = ibd.drop(columns=['Disease_Name'])
# y = ibd['Disease Name']
#
# #划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1, random_state=42)

#取prieb1220数据集为训练集，R语言包中数据集为测试集
ibd = pd.read_csv('./data/prieb1220_dataset.csv')
X_train = ibd.drop(columns=['Disease_Name'])
y_train = ibd['Disease_Name']

ibd = pd.read_csv('./data/r_dataset.csv')
X_test = ibd.drop(columns=['Disease_Name'])
y_test = ibd['Disease_Name']

training_accuracy = []
test_accuracy = []
ibd_features = ibd.drop(columns=['Disease_Name']).columns
# print(ibd.columns)
#
# #数据集维度，多少数据点，每个数据点有多少特征
# print("dimension of IBD data: {}".format(ibd.shape))
#
# #目标值（列名）对应的值和个数
# print(ibd.groupby('Disease Name').size())
#
#
# sns.countplot(ibd['Disease Name'],label="Count")
# ibd.info()

# K-NN算法预测ibd
#n_neighbors 从1到10
# neighbors_settings = range(1, 11)
# for n_neighbors in neighbors_settings:
#     #建立模型
#     knn = KNeighborsClassifier(n_neighbors=n_neighbors)
#     knn.fit(X_train, y_train)
#     # 训练集准确度
#     training_accuracy.append(knn.score(X_train, y_train))
#     # 测试集准确度
#     test_accuracy.append(knn.score(X_test, y_test))
# #绘制训练集和测试集准确性
# plt.plot(neighbors_settings, training_accuracy, label="训练集准确度")
# plt.plot(neighbors_settings, test_accuracy, label="测试集准确度")
# plt.ylabel("准确度")
# plt.xlabel("K近邻算法参数k")
# plt.legend()
# plt.show()

#检查k近邻算法预测ibd的准确性得分
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
# Accuracy of K-NN classifier on training set
# Accuracy of K-NN classifier on test set
print('训练集准确度: {:.2f}'.format(knn.score(X_train, y_train)))
print('测试集准确度: {:.2f}'.format(knn.score(X_test, y_test)))