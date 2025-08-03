import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             precision_score, recall_score, f1_score,
                             RocCurveDisplay, roc_curve, make_scorer)
import joblib

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示支持

# ==================== 数据配置 ====================
DATASETS = {
    "RF": "../train/rf_train.csv",
    "MI": "../train/mi_train.csv",
    "Autoencoder": "../train/Autoencoder_train.csv"
}


# ==================== 核心函数 ====================
def xgb_full_pipeline(data_type):
    """全流程建模评估函数"""
    # ---------- 数据加载 ----------
    train_data = pd.read_csv(DATASETS[data_type])
    X = train_data.drop('Disease', axis=1)
    y = train_data['Disease']

    # ---------- 网格搜索调参 ----------
    param_grid = {
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.15],
        'n_estimators': [20,30,50,100,150,200],
        'gamma': [0,0.1,0.2,0.3,0.4],
        'subsample':[0.7,0.8,0.9,0.95,1.0],
        'colsample_bytree': [0.1,0.2,0.3,0.4]
    }

    model = xgb.XGBClassifier(objective='binary:logistic',eval_metric='logloss',random_state=2025)

    searcher = GridSearchCV(model, param_grid,scoring='roc_auc',cv=3, n_jobs=-1)
    searcher.fit(X, y)

    # ---------- 保存最佳模型 ----------
    best_model = searcher.best_estimator_
    joblib.dump(best_model, f'xgb_best_{data_type}.pkl')

    # ---------- 五折交叉验证 ----------
    metrics = {
        'auc': 'roc_auc',
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score)
    }

    cv_results = cross_validate(best_model, X, y,cv=5, scoring=metrics,return_train_score=False,return_estimator=True)

    # ---------- 结果记录 ----------
    print(f"\n {data_type}数据集最优参数:")
    print(pd.DataFrame([searcher.best_params_]))

    print(f"\n 五折交叉验证结果:")
    results_df = pd.DataFrame({
        'Fold': [f'Fold {i + 1}' for i in range(5)],
        'AUC': cv_results['test_auc'],
        'Accuracy': cv_results['test_accuracy'],
        'Precision': cv_results['test_precision'],
        'Recall': cv_results['test_recall'],
        'F1': cv_results['test_f1']
    })
    print(results_df.round(4))

    print(f"\n 平均指标:")
    print(results_df.mean(numeric_only=True).round(4))


    # 绘制平均曲线
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
    """绘制双数据集平均ROC曲线"""
    plt.figure(figsize=(10, 6))
    colors = {'RF': '#FF6F61', 'MI': '#6B5B95'}

    # 绘制RF数据集曲线
    plt.plot(rf_data['mean_fpr'], rf_data['mean_tpr'],
             color=colors['RF'],
             label=f'(RF)XGBoost(AUC={rf_data["auc"]:.3f})',
             lw=2.5, linestyle='-')

    # 绘制MI数据集曲线
    plt.plot(mi_data['mean_fpr'], mi_data['mean_tpr'],
             color=colors['MI'],
             label=f'(MI-VAR)XGBoost(AUC={mi_data["auc"]:.3f})',
             lw=2.5, linestyle='--')

    # 公共样式设置
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
    plt.xlabel('FalsePositiveRate', fontsize=12)
    plt.ylabel('TruePositiveRate', fontsize=12)
    plt.title('XGBoost ROC curve', fontsize=14)
    plt.legend(loc='lower right')
    plt.savefig('combined_roc.png', dpi=300, bbox_inches='tight')
    plt.close()


# ==================== 执行入口 ====================
if __name__ == "__main__":
    rf_data = xgb_full_pipeline("RF")
    mi_data = xgb_full_pipeline("MI")
    auto_data = xgb_full_pipeline("Autoencoder")
    # plot_combined_roc(rf_data, mi_data)