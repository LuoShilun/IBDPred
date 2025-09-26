import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# 数据加载
train_data = pd.read_csv("../NewExpriment/r.csv")
test_data = pd.read_csv("../NewExpriment/hab.csv")

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

# 定义编码器
input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(128, activation='relu')(input_layer)
encoded = layers.Dense(64, activation='relu')(encoded)
encoded = layers.Dense(encoding_dim, activation='relu')(encoded)

# 定义解码器
decoded = layers.Dense(64, activation='relu')(encoded)
decoded = layers.Dense(128, activation='relu')(decoded)
decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)

# 构建自编码器模型
autoencoder = models.Model(input_layer, decoded)

# 编译模型
autoencoder.compile(optimizer='adam', loss='mse')

# 训练自编码器
history = autoencoder.fit(X_train_scaled, X_train_scaled, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

# 提取编码器部分
encoder = models.Model(input_layer, encoded)

# 使用编码器提取特征
encoded_train_data = encoder.predict(X_train_scaled)
encoded_test_data = encoder.predict(X_test_scaled)

# 将提取的特征转换为 DataFrame
encoded_train_df = pd.DataFrame(encoded_train_data, index=X_train.index)
encoded_test_df = pd.DataFrame(encoded_test_data, index=X_test.index)

# 选择最重要的30个特征
# 为了选择最重要的30个特征，我们需要分析编码器的权重矩阵
# 获取编码器的第一层权重矩阵
encoder_weights = encoder.layers[1].get_weights()[0]

# 计算每个特征的重要性（可以使用权重的绝对值之和作为重要性指标）
feature_importance = encoder_weights.sum(axis=0)

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
train_important_dataset.to_csv("../dataset/数据处理/实验4/train/Autoencoder_train.csv", index=False)
test_important_dataset.to_csv("../dataset/数据处理/实验4/test/Autoencoder_test.csv", index=False)