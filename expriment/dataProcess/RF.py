import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

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

# 方法1：随机森林特征重要性筛选
# 基于OOB（袋外评分）准确率筛选重要性Top30菌群
rf_selector = SelectFromModel(
    RandomForestClassifier(
        n_estimators=500,
        class_weight='balanced',
        oob_score=True  # 使用袋外估计避免过拟合
    ),
    max_features=30,
    threshold='1.5*median'  # 动态阈值
)
X_train_rf = rf_selector.fit_transform(X_train, y_train)

# 获取筛选特征的索引
selected_indices = rf_selector.get_support(indices=True)

# 获取原始特征名称
selected_features_rf = [original_feature_names[i] for i in selected_indices]

# 获取特征重要性
importances = rf_selector.estimator_.feature_importances_

# 创建特征重要性数据框
importance_df = pd.DataFrame({
    'Feature': selected_features_rf,
    'Importance': importances[selected_indices]
}).sort_values('Importance', ascending=False)

print("随机森林筛选特征TOP30:\n", importance_df.head(50))


# 设置全局字体大小
plt.rcParams['font.size'] = 12  # 默认字体大小
plt.rcParams['axes.labelsize'] = 14  # 坐标轴标签字体大小
plt.rcParams['axes.titlesize'] = 15  # 标题字体大小
plt.rcParams['xtick.labelsize'] = 12  # x轴刻度标签字体大小
plt.rcParams['ytick.labelsize'] = 12  # y轴刻度标签字体大小
# 可视化特征重要性（横向条形图）
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', legend=False)
plt.title('Feature Importance Ranking (Method RF)')
plt.tight_layout()
plt.show()

# 相关系数热力图
corr_matrix = pd.DataFrame(X_train_rf, columns=selected_features_rf).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', xticklabels=True, yticklabels=True)
plt.title('Feature Correlation Heatmap (Method RF)')
plt.show()

# 生成新数据集
X_train_rf = pd.DataFrame(rf_selector.transform(X_train), columns=selected_features_rf)
X_test_rf = pd.DataFrame(rf_selector.transform(X_test), columns=selected_features_rf)

# 合并特征与标签
train_rf_dataset = pd.concat([X_train_rf, y_train.reset_index(drop=True)], axis=1)
test_rf_dataset = pd.concat([X_test_rf, y_test.reset_index(drop=True)], axis=1)

# 保存数据集
train_rf_dataset.to_csv("../dataset/数据处理/实验4/train/rf_train.csv", index=False)
test_rf_dataset.to_csv("../dataset/数据处理/实验4/test/rf_test.csv", index=False)


# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
#
# # 数据加载
# train_data = pd.read_csv("../dataset/数据处理/实验4/r.csv")
# test_data = pd.read_csv("../dataset/数据处理/实验4/hab.csv")
#
# # 分离特征和标签
# XTrain = train_data.drop('Disease', axis=1)
# y_train = train_data['Disease']
# XTest = test_data.drop('Disease', axis=1)
# y_test = test_data['Disease']
#
# # 保存原始特征名称
# original_feature_names = XTrain.columns.tolist()
#
# # 数据标准化
# scaler = StandardScaler()
# X_train = scaler.fit_transform(XTrain)
# X_test = scaler.transform(XTest)
#
# # 提取之前筛选的特征名称
# selected_features = [
#     "Dorea_formicigenerans", "Anaerobutyricum_hallii", "Gemmiger_formicilis",
#     "Bifidobacterium_bifidum", "Mogibacterium_diversum",
#     "Agathobaculum_butyriciproducens", "Gordonibacter_pamelaeae",
#     "Collinsella_aerofaciens", "Blautia_hydrogenotrophica",
#     "Bifidobacterium_longum", "Bacteroides_xylanisolvens",
#     "Coprococcus_eutactus", "Coprococcus_comes", "Eubacterium_ramulus",
#     "Odoribacter_splanchnicus", "Fusicatenibacter_saccharivorans",
#     "Flavonifractor_plautii", "Ruminococcus_bromii",
#     "Anaerostipes_hadrus", "Streptococcus_thermophilus",
#     "Eggerthella_lenta", "Roseburia_inulinivorans",
#     "Faecalibacterium_prausnitzii", "Bifidobacterium_adolescentis",
#     "Enterocloster_citroniae", "Blautia_obeum", "Dorea_longicatena",
#     "Parabacteroides_distasonis", "Lachnospira_pectinoschiza",
#     "Alistipes_shahii"
# ]
#
# # 获取这些特征在原始数据中的索引
# selected_indices = [original_feature_names.index(feature) for feature in selected_features]
#
# # 根据索引筛选特征
# X_train_rf = X_train[:, selected_indices]
# X_test_rf = X_test[:, selected_indices]
#
# # 创建特征重要性数据框（假设重要性值与之前相同）
# importances = [
#     0.033160, 0.030009, 0.026536, 0.024809, 0.023415, 0.021794, 0.019646,
#     0.019280, 0.019108, 0.017775, 0.017616, 0.017474, 0.017073, 0.016715,
#     0.016435, 0.015909, 0.014592, 0.014072, 0.012993, 0.012657, 0.011367,
#     0.011307, 0.011117, 0.010962, 0.010584, 0.010484, 0.009476, 0.009262,
#     0.009245, 0.009108
# ]
#
# importance_df = pd.DataFrame({
#     'Feature': selected_features,
#     'Importance': importances
# }).sort_values('Importance', ascending=False)
#
# print("随机森林筛选特征TOP30:\n", importance_df.head(50))
#
# # 设置全局字体大小
# plt.rcParams['font.size'] = 12  # 默认字体大小
# plt.rcParams['axes.labelsize'] = 14  # 坐标轴标签字体大小
# plt.rcParams['axes.titlesize'] = 15  # 标题字体大小
# plt.rcParams['xtick.labelsize'] = 12  # x轴刻度标签字体大小
# plt.rcParams['ytick.labelsize'] = 12  # y轴刻度标签字体大小
#
# # 可视化特征重要性（横向条形图）
# plt.figure(figsize=(10, 8))
# sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', legend=False)
# plt.title('Feature Importance Ranking (Method RF)')
# plt.tight_layout()
# plt.show()
#
# # 相关系数热力图
# corr_matrix = pd.DataFrame(X_train_rf, columns=selected_features).corr()
# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', xticklabels=True, yticklabels=True)
# plt.title('Feature Correlation Heatmap (Method RF)')
# plt.show()
#
# # 生成新数据集
# X_train_rf = pd.DataFrame(X_train_rf, columns=selected_features)
# X_test_rf = pd.DataFrame(X_test_rf, columns=selected_features)
#
# # 合并特征与标签
# train_rf_dataset = pd.concat([X_train_rf, y_train.reset_index(drop=True)], axis=1)
# test_rf_dataset = pd.concat([X_test_rf, y_test.reset_index(drop=True)], axis=1)
#
# # 保存数据集
# train_rf_dataset.to_csv("../dataset/数据处理/实验4/train/rf_train.csv", index=False)
# test_rf_dataset.to_csv("../dataset/数据处理/实验4/test/rf_test.csv", index=False)