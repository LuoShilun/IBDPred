import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import numpy as np

# 数据加载
train_data = pd.read_csv(".../NewExpriment/数据处理/实验4/train/r.csv")
test_data = pd.read_csv(".../NewExpriment/数据处理/实验4/test/hab.csv")

# 分离特征和标签
X_train = train_data.drop('Disease', axis=1)
y_train = train_data['Disease']
X_test = test_data.drop('Disease', axis=1)
y_test = test_data['Disease']

# 保存原始特征名称
original_feature_names = X_train.columns.tolist()

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

input_dim = X_train_scaled.shape[1]  # 输入特征的维度
encoding_dim = 30  # 希望提取的特征数量

# 定义 MLP 模型
mlp = MLPRegressor(hidden_layer_sizes=(128, 64, encoding_dim), activation='relu', max_iter=500, random_state=42)

# 训练 MLP 模型
mlp.fit(X_train_scaled, y_train)

# 使用 MLP 提取特征
encoded_train_data = mlp.predict(X_train_scaled)
encoded_test_data = mlp.predict(X_test_scaled)

# 将提取的特征转换为 DataFrame
encoded_train_df = pd.DataFrame(encoded_train_data, index=X_train.index)
encoded_test_df = pd.DataFrame(encoded_test_data, index=X_test.index)

# 获取 MLP 的第一层权重矩阵
mlp_weights = mlp.coefs_[0]

# 计算每个特征的重要性（可以使用权重的绝对值之和作为重要性指标）
feature_importance = np.sum(np.abs(mlp_weights), axis=0)

# 获取最重要的30个特征的索引
top_indices = feature_importance.argsort()[-encoding_dim:][::-1]

# 获取最重要的30个特征的名字
important_feature_names = [original_feature_names[i] for i in top_indices]

# 获取最重要的30个特征的数据
important_train_features_data = X_train.iloc[:, top_indices]
important_test_features_data = X_test.iloc[:, top_indices]

# 将提取的特征和名字保存到新的 DataFrame
important_train_features_df = pd.DataFrame(important_train_features_data, columns=important_feature_names)
important_test_features_df = pd.DataFrame(important_test_features_data, columns=important_feature_names)

# 合并特征与标签
train_important_dataset = pd.concat([important_train_features_df, y_train.reset_index(drop=True)], axis=1)
test_important_dataset = pd.concat([important_test_features_df, y_test.reset_index(drop=True)], axis=1)

# 保存到新的 CSV 文件
train_important_dataset.to_csv("../dataset/数据处理/实验4/train/MLP_train.csv", index=False)
test_important_dataset.to_csv("../dataset/数据处理/实验4/test/MLP_test.csv", index=False)