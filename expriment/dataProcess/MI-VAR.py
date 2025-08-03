import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.preprocessing import PowerTransformer, StandardScaler

# 数据加载
train_data = pd.read_csv("../NewExpriment/数据处理/实验4/r.csv")
test_data = pd.read_csv("../NewExpriment/数据处理/实验4/hab.csv")

# 分离特征和标签
XTrain = train_data.drop('Disease', axis=1)
y_train = train_data['Disease']
XTest = test_data.drop('Disease', axis=1)
y_test = test_data['Disease']

# 保存原始特征名称
original_feature_names = XTrain.columns.tolist()

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(XTrain)
X_test = scaler.transform(XTest)

# 方法2：互信息+方差联合筛选
# 步骤1：方差过滤（去除低波动菌群）
pt = PowerTransformer()  # Yeo-Johnson变换
X_train_trans = pt.fit_transform(X_train)
selector_var = VarianceThreshold(threshold=0.02)  # 保留方差大于0.02的特征
X_train_highvar = selector_var.fit_transform(X_train_trans)

# 获取经过方差过滤后的特征索引
var_indices = selector_var.get_support(indices=True)

# 获取经过方差过滤后的特征名
var_features = [original_feature_names[i] for i in var_indices]

# 步骤2：互信息筛选Top30
mi_selector = SelectKBest(mutual_info_classif, k=30)
X_train_mi = mi_selector.fit_transform(X_train_highvar, y_train)

# 获取互信息筛选后的特征索引
mi_indices = mi_selector.get_support(indices=True)

# 获取最终筛选后的特征名称
selected_features_mi = [var_features[i] for i in mi_indices]

print(" 互信息筛选特征示例:\n", selected_features_mi)

# 相关系数热力图
corr_matrix_mi = pd.DataFrame(X_train_mi, columns=selected_features_mi).corr()
plt.rcParams['axes.titlesize'] = 30  # 标题字体大小
plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix_mi, cmap='viridis', annot=False, center=0,
            xticklabels=True, yticklabels=True)
plt.title('Feature Correlation Heatmap (MI-VAR)')
plt.show()

# 生成新数据集（需保持特征顺序一致）
X_train_mi = pd.DataFrame(mi_selector.transform(X_train_highvar), columns=selected_features_mi)
X_test_mi = pd.DataFrame(mi_selector.transform(selector_var.transform(pt.transform(X_test))), columns=selected_features_mi)

# 合并特征与标签
train_mi_dataset = pd.concat([X_train_mi, y_train.reset_index(drop=True)], axis=1)
test_mi_dataset = pd.concat([X_test_mi, y_test.reset_index(drop=True)], axis=1)

# 保存数据集
train_mi_dataset.to_csv("../dataset/数据处理/实验4/train/mi_train.csv", index=False)
test_mi_dataset.to_csv("../dataset/数据处理/实验4/test/mi_test.csv", index=False)