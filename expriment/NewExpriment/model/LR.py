# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import GridSearchCV, cross_validate
# from sklearn.metrics import (
#     roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
#     make_scorer, roc_curve, RocCurveDisplay
# )
#
# import joblib
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
#
# # ==================== 数据配置 ====================
# DATASETS = {
#     "RF": "../dataset/train/rf_train.csv",
#     "MI": "../dataset/train/mi_train.csv"
# }
#
# # ==================== 核心函数 ====================
# def lr_full_pipeline(data_type):
#     """全流程建模评估函数"""
#     # ---------- 数据加载 ----------
#     train_data = pd.read_csv(DATASETS[data_type])
#     X = train_data.drop('Disease', axis=1)
#     y = train_data['Disease']
#
#     # ---------- 网格搜索调参 ----------
#     param_grid = {
#         'C': [0.001,0.01, 0.1, 1, 10, 100],  # 正则化强度的倒数
#         'solver': ['liblinear', 'saga'],  # 求解器
#         'penalty': ['l1', 'l2']  # 正则化类型
#     }
#
#     model = LogisticRegression(random_state=2025, max_iter=2000)  # 增加最大迭代次数
#
#     searcher = GridSearchCV(model, param_grid, scoring='roc_auc', cv=3, n_jobs=-1)
#     searcher.fit(X, y)
#
#     # ---------- 保存最佳模型 ----------
#     best_model = searcher.best_estimator_
#     joblib.dump(best_model, f'./ModelPKL/lr_best_{data_type}.pkl')
#
#     # ---------- 五折交叉验证 ----------
#     metrics = {
#         'auc': 'roc_auc',
#         'accuracy': make_scorer(accuracy_score),
#         'precision': make_scorer(precision_score),
#         'recall': make_scorer(recall_score),
#         'f1': make_scorer(f1_score)
#     }
#
#     cv_results = cross_validate(best_model, X, y, cv=5, scoring=metrics,
#                                 return_train_score=False, return_estimator=True)
#
#     # ---------- 结果记录 ----------
#     print(f"\n {data_type}数据集最优参数:")
#     print(pd.DataFrame([searcher.best_params_]))
#
#     print(f"\n 五折交叉验证结果:")
#     results_df = pd.DataFrame({
#         'Fold': [f'Fold {i + 1}' for i in range(5)],
#         'AUC': cv_results['test_auc'],
#         'Accuracy': cv_results['test_accuracy'],
#         'Precision': cv_results['test_precision'],
#         'Recall': cv_results['test_recall'],
#         'F1': cv_results['test_f1']
#     })
#     print(results_df.round(4))
#
#     print(f"\n 平均指标:")
#     print(results_df.mean(numeric_only=True).round(4))
#
#     # 绘制平均曲线
#     mean_fpr = np.linspace(0, 1, 100)
#     tprs = []
#     for estimator in cv_results['estimator']:
#         pred_proba = estimator.predict_proba(X)[:, 1]
#         fpr, tpr, _ = roc_curve(y, pred_proba)
#         tprs.append(np.interp(mean_fpr, fpr, tpr))
#
#     mean_tpr = np.mean(tprs, axis=0)
#
#     return {
#         'mean_fpr': mean_fpr,
#         'mean_tpr': mean_tpr,
#         'auc': np.mean(cv_results["test_auc"])
#     }
#
# # ==================== 新增绘图函数 ====================
# def plot_combined_roc(rf_data, mi_data):
#     """绘制双数据集平均ROC曲线"""
#     plt.figure(figsize=(10, 6))
#     colors = {'RF': '#FF6F61', 'MI': '#6B5B95'}
#
#     # 绘制RF数据集曲线
#     plt.plot(rf_data['mean_fpr'], rf_data['mean_tpr'],
#              color=colors['RF'],
#              label=f'(RF)LR(AUC={rf_data["auc"]:.3f})',
#              lw=2.5, linestyle='-')
#
#     # 绘制MI数据集曲线
#     plt.plot(mi_data['mean_fpr'], mi_data['mean_tpr'],
#              color=colors['MI'],
#              label=f'(MI-VAR)LR(AUC={mi_data["auc"]:.3f})',
#              lw=2.5, linestyle='--')
#
#     # 公共样式设置
#     plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
#     plt.xlabel('False Positive Rate', fontsize=12)
#     plt.ylabel('True Positive Rate', fontsize=12)
#     plt.title('LR ROC Curve', fontsize=14)
#     plt.legend(loc='lower right')
#     plt.savefig(',/resultsImage/combined_roc_lr.png', dpi=300, bbox_inches='tight')
#     plt.close()
#
# # ==================== 执行入口 ====================
# if __name__ == "__main__":
#     rf_data = lr_full_pipeline("RF")
#     mi_data = lr_full_pipeline("MI")
#     plot_combined_roc(rf_data, mi_data)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, RocCurveDisplay, make_scorer
)
import joblib
import os

plt.rcParams['font.sans-serif'] = ['SimHei']

# ==================== 数据配置 ====================
DATASETS = {
    "RF": "../dataset/train/rf_train.csv",
    "MI": "../dataset/train/mi_train.csv",
    "Autoencoder": "../dataset/train/Autoencoder_train.csv"
}

# ==================== 核心函数 ====================
def lr_full_pipeline(data_type):
    """全流程建模评估函数"""
    # ---------- 数据加载 ----------
    train_data = pd.read_csv(DATASETS[data_type])
    X = train_data.drop('Disease', axis=1)
    y = train_data['Disease']

    # ---------- 直接使用默认参数的逻辑回归 ----------
    model = LogisticRegression(random_state=2025)

    # ---------- 五折交叉验证 ----------
    metrics = {
        'auc': 'roc_auc',
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score)
    }
    cv_results = cross_validate(model, X, y, cv=5, scoring=metrics,
                                return_train_score=False, return_estimator=True)

    # ---------- 保存模型 ----------
    model_path = f'./ModelPKL/lr_{data_type}.pkl'
    joblib.dump(model, model_path)
    print(f"模型已保存到 {model_path}")

    # 检查文件大小
    file_size = os.path.getsize(model_path)
    if file_size > 0:
        print(f"文件大小：{file_size} 字节")
    else:
        print("模型文件保存失败，文件大小为 0")

    # ---------- 结果记录 ----------
    print(f"\n {data_type} 数据集的交叉验证结果：")
    results_df = pd.DataFrame({
        'Fold': [f'Fold {i + 1}' for i in range(5)],
        'AUC': cv_results['test_auc'],
        'Accuracy': cv_results['test_accuracy'],
        'Precision': cv_results['test_precision'],
        'Recall': cv_results['test_recall'],
        'F1': cv_results['test_f1']
    })
    print(results_df.round(4))

    print(f"\n {data_type} 数据集的平均指标：")
    print(results_df.mean(numeric_only=True).round(4))

    # 绘制平均 ROC 曲线
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    for estimator in cv_results['estimator']:
        pred_proba = estimator.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, pred_proba)
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)

    return {
        'mean_fpr': mean_fpr,
        'mean_tpr': mean_tpr,
        'auc': np.mean(cv_results["test_auc"])
    }

# ==================== 新增绘图函数 ====================
def plot_combined_roc(rf_data, mi_data):
    """绘制双数据集平均 ROC 曲线"""
    plt.figure(figsize=(10, 6))
    colors = {'RF': '#FF6F61', 'MI': '#6B5B95'}

    # 绘制 RF 数据集曲线
    plt.plot(rf_data['mean_fpr'], rf_data['mean_tpr'],
             color=colors['RF'],
             label=f'(RF) LR (AUC={rf_data["auc"]:.3f})',
             lw=2.5, linestyle='-')

    # 绘制 MI 数据集曲线
    plt.plot(mi_data['mean_fpr'], mi_data['mean_tpr'],
             color=colors['MI'],
             label=f'(MI-VAR) LR (AUC={mi_data["auc"]:.3f})',
             lw=2.5, linestyle='--')

    # 公共样式设置
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('LR ROC Curve', fontsize=14)
    plt.legend(loc='lower right')
    plt.savefig('./resultsImage/combined_roc_lr.png', dpi=300, bbox_inches='tight')
    plt.close()

# ==================== 执行入口 ====================
if __name__ == "__main__":
    rf_data = lr_full_pipeline("RF")
    mi_data = lr_full_pipeline("MI")
    auto_data = lr_full_pipeline("Autoencoder")
    # plot_combined_roc(rf_data, mi_data)