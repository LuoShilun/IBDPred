import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score, mean_squared_error, make_scorer, roc_curve
from sklearn.model_selection import cross_validate, KFold, cross_val_predict, StratifiedKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin


# ================= 数据加载器 =================
def load_features(method):
    """加载预筛选特征数据集"""
    return (
        pd.read_csv(f"../dataset/train/{method}_train.csv"),
        pd.read_csv(f"../dataset/test/{method}_test.csv")
    )


# ================= 加权堆叠分类器 =================
class WeightedStackingClassifier(BaseEstimator, ClassifierMixin):
    """
    基于精确度权重和多样性权重的堆叠分类器

    参数:
    - estimators: 基学习器列表，格式为[(name, estimator), ...]
    - alpha: 精确性权重和多样性权重之间的平衡参数，0.5
    - cv: 交叉验证折数
    """

    def __init__(self, estimators, alpha=0.7, cv=5):
        self.estimators = estimators
        self.alpha = alpha
        self.cv = cv
        self.estimator_weights_ = None
        self.base_estimators_ = {}
        self.classes_ = None

    def fit(self, X, y):
        """训练模型并计算基学习器权重"""
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_estimators = len(self.estimators)

        # 初始化存储交叉验证预测结果的数组
        cv_predictions = np.zeros((X.shape[0], n_estimators, n_classes))

        # 训练每个基学习器并获取交叉验证预测
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)

        for i, (name, estimator) in enumerate(self.estimators):
            print(f"训练基学习器: {name}")
            self.base_estimators_[name] = estimator.fit(X, y)

            # 获取交叉验证预测
            fold_idx = 0
            for train_idx, val_idx in kf.split(X):
                if isinstance(X, np.ndarray):
                    X = pd.DataFrame(X)
                if isinstance(y, np.ndarray):
                    y = pd.Series(y)

                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                for train_idx, val_idx in skf.split(X, y):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                # 训练模型并预测概率
                est = estimator.__class__(**estimator.get_params())
                est.fit(X_train, y_train)
                proba = est.predict_proba(X_val)

                # 存储预测结果
                cv_predictions[val_idx, i, :] = proba
                fold_idx += 1

        # 计算精确性权重 (基于RMSE)
        accuracy_weights = np.zeros(n_estimators)
        for i in range(n_estimators):
            # 将概率预测转换为类别预测
            y_pred = np.argmax(cv_predictions[:, i, :], axis=1)

            # 计算RMSE (对于分类问题，使用0-1损失的均方根误差)
            rmse = np.sqrt(mean_squared_error(y, y_pred))

            # 精确性权重与RMSE的倒数成正比
            accuracy_weights[i] = 1.0 / (rmse + 1e-10)  # 添加小值避免除零

        # 归一化精确性权重
        accuracy_weights = accuracy_weights / np.sum(accuracy_weights)

        # 计算多样性权重 (基于预测结果的相关性)
        diversity_weights = np.zeros(n_estimators)
        for i in range(n_estimators):
            diversity_sum = 0
            for j in range(n_estimators):
                if i != j:
                    # 计算两个模型预测结果的相关系数
                    pred_i = np.argmax(cv_predictions[:, i, :], axis=1)
                    pred_j = np.argmax(cv_predictions[:, j, :], axis=1)

                    # 计算相关系数 (对于分类问题，使用预测是否一致作为相关性度量)
                    agreement = np.mean(pred_i == pred_j)
                    correlation = 2 * agreement - 1  # 将[0,1]映射到[-1,1]

                    # 多样性与相关系数的绝对值成反比
                    diversity_sum += (1 - abs(correlation))

            # 平均多样性
            diversity_weights[i] = diversity_sum / (n_estimators - 1) if n_estimators > 1 else 0

        # 归一化多样性权重
        if np.sum(diversity_weights) > 0:
            diversity_weights = diversity_weights / np.sum(diversity_weights)

        # 结合精确性权重和多样性权重，计算混合权重
        mixed_weights = self.alpha * accuracy_weights + (1 - self.alpha) * diversity_weights

        # 归一化混合权重
        self.estimator_weights_ = mixed_weights / np.sum(mixed_weights)

        # 输出每个模型的权重
        for i, (name, _) in enumerate(self.estimators):
            print(f"模型 {name} 的权重: {self.estimator_weights_[i]:.4f} "
                  f"(精确性权重: {accuracy_weights[i]:.4f}, 多样性权重: {diversity_weights[i]:.4f})")

        return self

    def predict_proba(self, X):
        """使用加权投票进行概率预测"""
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        # 初始化概率矩阵
        proba = np.zeros((n_samples, n_classes))

        # 加权组合每个基学习器的预测
        for i, (name, _) in enumerate(self.estimators):
            estimator = self.base_estimators_[name]
            proba += self.estimator_weights_[i] * estimator.predict_proba(X)

        return proba

    def predict(self, X):
        """预测类别"""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# ================= 模型验证器 =================
def evaluate_model(model, train_data, test_data):
    """执行模型训练与指标输出"""
    # 数据拆分
    X_train = train_data.drop('Disease', axis=1)
    y_train = train_data['Disease']
    X_test = test_data.drop('Disease', axis=1)
    y_test = test_data['Disease']

    metrics = {
        'auc': 'roc_auc',
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score)
    }

    cv_results = cross_validate(model, X_train, y_train, cv=5, scoring=metrics,
                                return_train_score=False, return_estimator=True)
    results_df = pd.DataFrame({
        'Fold': [f'Fold {i + 1}' for i in range(5)],
        'AUC': cv_results['test_auc'],
        'Accuracy': cv_results['test_accuracy'],
        'Precision': cv_results['test_precision'],
        'Recall': cv_results['test_recall'],
        'F1': cv_results['test_f1']
    })
    print("\n" + "=" * 50)
    print("训练集上的评价指标")
    print(results_df.mean(numeric_only=True).round(4))
    # 绘制平均曲线
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    for estimator in cv_results['estimator']:
        pred_proba = estimator.predict_proba(X_train)[:, 1]
        fpr, tpr, _ = roc_curve(y_train, pred_proba)
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)


    # 模型训练
    model.fit(X_train, y_train)

    # 在测试集上评估模型性能
    y_pred2 = model.predict(X_test)
    y_prob2 = model.predict_proba(X_test)[:, 1]  # 获取预测概率

    print("\n" + "=" * 50)
    print("测试集上的评价指标")
    print(f"Accuracy: {accuracy_score(y_test, y_pred2):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred2):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred2):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred2):.4f}")
    print(f"AUC: {roc_auc_score(y_test, y_prob2):.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred2))
    print("\nClassification Report:\n", classification_report(y_test, y_pred2))
    print("=" * 50 + "\n")

    # 保存训练好的模型
    joblib.dump(model, f"./ModelPKL/model_new_{feature_method}.pkl")
    print(f"模型已保存为 model_new_{feature_method}.pkl 文件")

    # 返回模型、训练数据、交叉验证预测概率和真实标签
    return {
        'mean_fpr': mean_fpr,
        'mean_tpr': mean_tpr,
        'auc': np.mean(cv_results["test_auc"])
    }
#alpha确定精确性权重的占比
# ================= 加入搜索alpha evaluate_model() =================
# def evaluate_model(model, train_data, test_data):
#     """执行模型训练与指标输出（含动态 α 搜索）"""
#     X_train = train_data.drop('Disease', axis=1)
#     y_train = train_data['Disease']
#     X_test  = test_data.drop('Disease', axis=1)
#     y_test  = test_data['Disease']
#
#     # ---------- 1. 在训练集上用交叉验证搜索最佳 α ----------
#     best_alpha, best_auc = None, -np.inf
#     alpha_grid = np.arange(0.1, 1.01, 0.1)
#     metrics = {'auc': 'roc_auc'}
#
#     for alpha in alpha_grid:
#         model.alpha = alpha                          # 动态设置 α
#         cv = cross_validate(model, X_train, y_train,
#                             cv=5, scoring=metrics,
#                             return_train_score=False)
#         mean_auc = cv['test_auc'].mean()
#         if mean_auc > best_auc:
#             best_auc, best_alpha = mean_auc, alpha
#
#     print(f"\n最佳 α = {best_alpha:.2f}, 对应 CV-AUC = {best_auc:.4f}")
#
#     # ---------- 2. 用最佳 α 重新训练并评估 ----------
#     model.alpha = best_alpha
#     cv_results = cross_validate(model, X_train, y_train, cv=5,
#                                 scoring={'auc': 'roc_auc'},
#                                 return_train_score=False,
#                                 return_estimator=True)
#
#     mean_fpr = np.linspace(0, 1, 100)
#     tprs = []
#     for est in cv_results['estimator']:
#         proba = est.predict_proba(X_train)[:, 1]
#         fpr, tpr, _ = roc_curve(y_train, proba)
#         tprs.append(np.interp(mean_fpr, fpr, tpr))
#     mean_tpr = np.mean(tprs, axis=0)
#
#     # 在全部训练集上最终训练
#     model.fit(X_train, y_train)
#
#     # 测试集表现
#     y_pred  = model.predict(X_test)
#     y_prob  = model.predict_proba(X_test)[:, 1]
#     print("\n测试集指标")
#     print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
#     print(f"Precision: {precision_score(y_test, y_pred):.4f}")
#     print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
#     print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
#     print(f"AUC      : {roc_auc_score(y_test, y_prob):.4f}")
#
#     joblib.dump(model, f"model_new_{feature_method}.pkl")
#     return {'mean_fpr': mean_fpr, 'mean_tpr': mean_tpr, 'auc': best_auc}

def plot_roc_curves(cv_results_dict):
    """绘制模型在不同数据集上的ROC曲线

    参数:
    - cv_results_dict: 字典，包含交叉验证结果，格式为：
      {
          'rf': {'mean_fpr': mean_fpr, 'mean_tpr': mean_tpr, 'auc': auc_value},
          'mi': {'mean_fpr': mean_fpr, 'mean_tpr': mean_tpr, 'auc': auc_value}
      }
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))

    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示支持
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6, label='随机猜测')

    # 颜色设置
    colors = {
        'rf': '#F44336',  # 红色
        'mi': '#2196F3'   # 蓝色
    }

    # 线型设置 - MI使用虚线
    line_styles = {
        'rf': '-',    # 实线
        'mi': '--'    # 虚线
    }

    # 绘制每个数据集上的ROC曲线
    for data_type, results in cv_results_dict.items():
        mean_fpr = results['mean_fpr']
        mean_tpr = results['mean_tpr']
        auc_value = results['auc']

        # 绘制ROC曲线
        plt.plot(
            mean_fpr, mean_tpr,
            color=colors[data_type],
            linestyle=line_styles[data_type],
            lw=2,
            label=f'改进Stacking-{data_type.upper()} (AUC = {auc_value:.4f})'
        )

    # 设置图形样式
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (False Positive Rate)', fontsize=12)
    plt.ylabel('真阳性率 (True Positive Rate)', fontsize=12)
    plt.title('改进Stacking模型在不同特征选择方法上的ROC曲线', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    # 添加坐标轴刻度
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))

    # 保存图形
    plt.savefig('./resultsImage/improved_stacking_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("ROC曲线图已保存为 improved_stacking_roc_curves.png")
    print("交叉验证AUC值：", {k: v['auc'] for k, v in cv_results_dict.items()})

# ================= 执行流程 =================
if __name__ == "__main__":
    # 存储交叉验证结果
    cv_results_dict = {}
    # 遍历两种特征筛选方法
    for feature_method in ['rf', 'mi']:
        # 加载数据
        train_set, test_set = load_features(feature_method)

        # 构建基学习器
        estimators = [
            ('svm', SVC(random_state=2025, C=10 if 'mi' in feature_method else 10,
                        kernel='rbf',
                        gamma=0.01 if 'mi' in feature_method else 0.01, probability=True)),
            ('xgb', XGBClassifier(random_state=2025, learning_rate=0.15 if 'mi' in feature_method else 0.05,
                                  max_depth=7 if 'mi' in feature_method else 3,
                                  gamma=0.1 if 'mi' in feature_method else 0,
                                  n_estimators=50 if 'mi' in feature_method else 250,
                                  subsample=0.9 if 'mi' in feature_method else 0.8,
                                  colsample_bytree=0.1 if 'mi' in feature_method else 0.4)),
            ('rf', RandomForestClassifier(random_state=2025, n_estimators=70 if 'mi' in feature_method else 20,
                                          max_depth=10 if 'mi' in feature_method else 14,
                                          min_samples_split=3 if 'mi' in feature_method else 6))
        ]

        # 使用加权堆叠分类器
        model = WeightedStackingClassifier(
            estimators=estimators,
            alpha=0.5 if 'mi' in feature_method else 0.7,  # 精确性权重比例
            cv=5
        )

        # 执行验证
        print(f"\n {feature_method.upper()}数据集评估结果 ")
        cv_results_dict[feature_method] = evaluate_model(model, train_set, test_set)


    # 调用绘制ROC曲线的函数
    plot_roc_curves(cv_results_dict)

    # 保存结果字典到文件
    import pickle
    with open('ModelPKL/stacking_improved_cv_results.pkl', 'wb') as f:
        pickle.dump(cv_results_dict, f)
    print("交叉验证结果已保存到 stacking_improved_cv_results.pkl")



