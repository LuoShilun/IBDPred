import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score, make_scorer, roc_curve
from sklearn.model_selection import cross_validate, KFold
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt


# ================= 数据加载器 =================
def load_features(method):
    """加载预筛选特征数据集"""
    return (
        pd.read_csv(f"../dataset/train/{method}_train.csv"),
        pd.read_csv(f"../dataset/test/{method}_test.csv")
    )


# ================= 加权平均集成分类器 =================
class WeightedAveragingEnsemble(BaseEstimator, ClassifierMixin):
    """
    基于加权平均的集成学习模型
    参数:
    - estimators: 基学习器列表，格式为[(name, estimator), ...]
    - weights: 各基学习器的权重，如果为None则自动计算
    - cv: 交叉验证折数，用于自动计算权重
    """
    
    def __init__(self, estimators, weights=None, cv=5):
        self.estimators = estimators
        self.weights = weights
        self.cv = cv
        self.base_estimators_ = {}
        self.classes_ = None
        self.weights_ = None
    
    def fit(self, X, y):
        """训练模型并计算基学习器权重"""
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_estimators = len(self.estimators)
        
        # 训练所有基学习器
        for name, estimator in self.estimators:
            self.base_estimators_[name] = estimator.fit(X, y)
        
        # 如果没有提供权重，则通过交叉验证自动计算
        if self.weights is None:
            print("自动计算基学习器权重...")
            
            # 初始化权重数组
            self.weights_ = np.zeros(n_estimators)
            
            # 使用K折交叉验证计算每个基学习器的性能
            kf = KFold(n_splits=self.cv, shuffle=True, random_state=2025)
            
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # 在训练集上训练每个基学习器
                for i, (name, estimator) in enumerate(self.estimators):
                    # 训练模型
                    estimator.fit(X_train, y_train)
                    
                    # 在验证集上评估性能
                    y_pred = estimator.predict(X_val)
                    
                    # 使用F1分数作为权重指标
                    score = f1_score(y_val, y_pred)
                    
                    # 累加权重
                    self.weights_[i] += score
            
            # 计算平均权重
            self.weights_ = self.weights_ / self.cv
            
            # 归一化权重
            self.weights_ = self.weights_ / np.sum(self.weights_)
        else:
            # 使用提供的权重
            self.weights_ = np.array(self.weights)
            
            # 确保权重和为1
            if np.sum(self.weights_) != 1.0:
                self.weights_ = self.weights_ / np.sum(self.weights_)
        
        # 输出每个模型的权重
        for i, (name, _) in enumerate(self.estimators):
            print(f"模型 {name} 的权重: {self.weights_[i]:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """使用加权平均进行概率预测"""
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # 初始化概率矩阵
        proba = np.zeros((n_samples, n_classes))
        
        # 对每个基学习器的预测结果进行加权平均
        for i, (name, _) in enumerate(self.estimators):
            estimator = self.base_estimators_[name]
            proba += self.weights_[i] * estimator.predict_proba(X)
        
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

    # 五折交叉验证
    metrics = {
        'auc': 'roc_auc',
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score)
    }

    cv_results = cross_validate(model, X_train, y_train, cv=5, scoring=metrics,
                                return_train_score=False, return_estimator=True)
    
    # 输出交叉验证结果
    results_df = pd.DataFrame({
        'Fold': [f'Fold {i + 1}' for i in range(5)],
        'AUC': cv_results['test_auc'],
        'Accuracy': cv_results['test_accuracy'],
        'Precision': cv_results['test_precision'],
        'Recall': cv_results['test_recall'],
        'F1': cv_results['test_f1']
    })
    print("\n" + "=" * 50)
    print("交叉验证结果（训练集）")
    print(results_df.round(4))
    print("\n平均指标:")
    print(results_df.mean(numeric_only=True).round(4))

    # 绘制平均ROC曲线
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    for estimator in cv_results['estimator']:
        pred_proba = estimator.predict_proba(X_train)[:, 1]
        fpr, tpr, _ = roc_curve(y_train, pred_proba)
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = np.mean(cv_results["test_auc"])

    # 模型训练
    model.fit(X_train, y_train)

    # 在测试集上评估模型性能
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # 获取预测概率

    print("\n" + "=" * 50)
    print("测试集上的评价指标")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("=" * 50 + "\n")

    # 保存训练好的模型
    joblib.dump(model, f"./ModelPKL/model_{feature_method}_weighted.pkl")
    print(f"模型已保存为 model_{feature_method}_weighted.pkl 文件")
    
    return {
        'mean_fpr': mean_fpr,
        'mean_tpr': mean_tpr,
        'auc': mean_auc
    }


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
                        gamma=0.01 if 'mi' in feature_method else 0.1, probability=True)),
            ('xgb', XGBClassifier(random_state=2025, learning_rate=0.15 if 'mi' in feature_method else 0.05,
                                  max_depth=7 if 'mi' in feature_method else 3,
                                  gamma=0.1 if 'mi' in feature_method else 0,
                                  n_estimators=50 if 'mi' in feature_method else 200,
                                  subsample=0.9 if 'mi' in feature_method else 0.8,
                                  colsample_bytree=0.1 if 'mi' in feature_method else 0.1)),
            ('rf', RandomForestClassifier(random_state=2025, n_estimators=70 if 'mi' in feature_method else 20,
                                          max_depth=10 if 'mi' in feature_method else 14,
                                          min_samples_split=3 if 'mi' in feature_method else 6)),
            # 添加逻辑回归模型
            ('lr', LogisticRegression(random_state=2025, C=0.01))
        ]

        # 使用加权平均集成学习模型
        model = WeightedAveragingEnsemble(
            estimators=estimators,
            weights=None,  # 自动计算权重
            cv=5
        )

        # 执行验证
        print(f"\n {feature_method.upper()}数据集评估结果 ")
        cv_results_dict[feature_method] = evaluate_model(model, train_set, test_set)
    
    # 保存结果字典到文件
    import pickle
    with open('ModelPKL/average_cv_results.pkl', 'wb') as f:
        pickle.dump(cv_results_dict, f)
    print("交叉验证结果已保存到 average_cv_results.pkl")
    # 绘制ROC曲线
    plt.figure(figsize=(10, 6))
    colors = {'rf': '#FF6F61', 'mi': '#6B5B95'}
        
    for method, result in cv_results_dict.items():
        plt.plot(result['mean_fpr'], result['mean_tpr'],
                 color=colors[method],
                 label=f'WeightedAverage-{method.upper()}(AUC={result["auc"]:.3f})',
                 lw=2.5, linestyle='-' if method == 'rf' else '--')
        
    # 公共样式设置
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Weighted Average ROC Curve', fontsize=14)
    plt.legend(loc='lower right')
    plt.savefig('./resultsImage/weighted_average_roc.png', dpi=300, bbox_inches='tight')
    plt.close()
        
    print("ROC曲线已保存为 weighted_average_roc.png")