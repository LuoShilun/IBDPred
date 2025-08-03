import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import os
# 添加以下导入语句
from Stacking改 import WeightedStackingClassifier
from average import WeightedAveragingEnsemble

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示支持
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ==================== 数据配置 ====================
DATASETS = {
    "rf": "../train/rf_train.csv",
    "mi": "../train/mi_train.csv"
}

# 测试集数据路径
TEST_DATASETS = {
    "rf": "../test/rf_test.csv",
    "mi": "../test/mi_test.csv"
}

# 模型文件路径 - 简化为三种模型在两个数据集上
MODEL_FILES = {
    "软投票-MI-VAR": {
        "mi": "voting_model_mi.pkl",
    },
    "软投票-RF": {
        "rf": "voting_model_rf.pkl",
    },
    "改进Stacking-RF": {
        "rf": "model_new_rf.pkl",
    },
    "改进Stacking-MI-VAR": {
        "mi": "model_new_mi.pkl",
    },
    "加权平均融合-MI-VAR": {
        "mi": "model_mi_weighted.pkl",
    },
    "加权平均融合-RF": {
        "rf": "model_rf_weighted.pkl",
    }
}

# 颜色和线型配置 - 与图片保持一致
COLORS = {
    "软投票-MI-VAR": "#4CAF50",  # 绿色
    "软投票-RF": "#4CAF50",
    "改进Stacking-MI-VAR": "#F44336",  # 红色
    "改进Stacking-RF": "#F44336",
    "加权平均融合-MI-VAR": "#2196F3",  # 蓝色
    "加权平均融合-RF": "#2196F3"
}

LINE_STYLES = {
    "软投票-MI-VAR": "-.",
    "软投票-RF": "--",
    "改进Stacking-MI-VAR": "-.",
    "改进Stacking-RF": "--",
    "加权平均融合-MI-VAR": "-.",
    "加权平均融合-RF": "-"
}

def load_data(data_type, is_test=False):
    """加载数据集"""
    if is_test:
        data_path = TEST_DATASETS[data_type]
    else:
        data_path = DATASETS[data_type]
    
    data = pd.read_csv(data_path)
    X = data.drop('Disease', axis=1)
    y = data['Disease']
    return X, y

def calculate_roc_data(model, X, y):
    """计算ROC曲线数据"""
    y_pred_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def plot_combined_roc():
    """绘制所有模型在一张图上的ROC曲线"""
    plt.figure(figsize=(10, 8))
    
    # 加载数据
    rf_X, rf_y = load_data("rf")
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k-', lw=1)
    
    # 存储模型和对应的曲线，用于后续添加图例
    lines = []
    labels = []
    
    # 绘制每个模型的ROC曲线
    for model_name in MODEL_FILES.keys():
        model_path = MODEL_FILES[model_name]["rf"]
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                # 计算ROC数据
                fpr, tpr, roc_auc = calculate_roc_data(model, rf_X, rf_y)
                # 格式化AUC值，保留5位小数
                auc_str = f"{roc_auc:.5f}"
                # 绘制ROC曲线
                line, = plt.plot(fpr, tpr, 
                         color=COLORS[model_name], 
                         linestyle=LINE_STYLES[model_name],
                         lw=2)
                lines.append(line)
                labels.append(f"{model_name} (AUC={auc_str})")
            except Exception as e:
                print(f"加载模型 {model_name} 时出错: {e}")
    # 设置图形样式
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率', fontsize=12)
    plt.ylabel('真阳性率', fontsize=12)
    plt.title('ROC (IBD)', fontsize=16)
    plt.grid(True, alpha=0.3)
    # 添加图例
    plt.legend(lines, labels, loc="lower right", fontsize=9)
    # 保存图形
    plt.savefig('roc_combined.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("ROC曲线图已保存为 roc_combined.png")

def plot_confusion_matrix():
    """绘制改进Stacking模型在RF筛选下的测试集上的混淆矩阵"""
    # 加载测试数据
    X_test, y_test = load_data("rf", is_test=True)
    
    # 加载改进Stacking-RF模型
    model_path = MODEL_FILES["改进Stacking-RF"]["rf"]
    
    if os.path.exists(model_path):
        try:
            # 加载模型
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)
            # 计算混淆矩阵
            cm = confusion_matrix(y_test, y_pred)
            # 绘制混淆矩阵
            plt.figure(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['health', 'IBD'])
            disp.plot(cmap='Blues', values_format='d')
            # 设置标题和标签
            plt.title('', fontsize=14)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            
            # 保存图形
            plt.savefig('stacking_rf_confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("混淆矩阵已保存为 stacking_rf_confusion_matrix.png")
            
            # 输出混淆矩阵的详细信息
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n混淆矩阵详细信息:")
            print(f"真阳性(TP): {tp}")
            print(f"假阳性(FP): {fp}")
            print(f"假阴性(FN): {fn}")
            print(f"真阴性(TN): {tn}")
            print(f"准确率(Accuracy): {accuracy:.4f}")
            print(f"精确率(Precision): {precision:.4f}")
            print(f"召回率(Recall): {recall:.4f}")
            print(f"特异性(Specificity): {specificity:.4f}")
            print(f"F1分数: {f1:.4f}")
            
        except Exception as e:
            print(f"绘制混淆矩阵时出错: {e}")
    else:
        print(f"模型文件 {model_path} 不存在")

def plot_all_models_roc():
    """绘制加权平均、软投票、stacking和改进stacking模型在mi和rf训练集上的ROC曲线图"""
    plt.figure(figsize=(12, 10))
    
    # 定义模型名称和对应的文件路径
    models = {
        "加权平均-RF": "model_rf_weighted.pkl",
        "加权平均-MI": "model_mi_weighted.pkl",
        "软投票-RF": "voting_model_rf.pkl",
        "软投票-MI": "voting_model_mi.pkl",
        "Stacking-RF": "model_rf.pkl",
        "Stacking-MI": "model_mi.pkl",
        "改进Stacking-RF": "model_new_rf.pkl",
        "改进Stacking-MI": "model_new_mi.pkl"
    }
    
    # 定义颜色和线型
    colors = {
        "加权平均": "#2196F3",  # 蓝色
        "软投票": "#4CAF50",    # 绿色
        "Stacking": "#FFC107",  # 黄色
        "改进Stacking": "#F44336"  # 红色
    }
    
    line_styles = {
        "RF": "-",   # 实线
        "MI": "--"   # 虚线
    }
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7)
    
    # 存储图例信息
    lines = []
    labels = []
    
    # 加载数据集
    rf_X, rf_y = load_data("rf")
    mi_X, mi_y = load_data("mi")
    
    # 遍历所有模型并绘制ROC曲线
    for model_name, model_file in models.items():
        try:
            # 确定数据集类型和基础模型类型
            dataset_type = model_name.split("-")[1]  # RF或MI
            base_model_type = model_name.split("-")[0]  # 加权平均、软投票等
            
            # 选择对应的数据集
            X = rf_X if dataset_type == "RF" else mi_X
            y = rf_y if dataset_type == "RF" else mi_y
            
            # 加载模型
            if os.path.exists(model_file):
                model = joblib.load(model_file)
                
                # 计算ROC曲线数据
                fpr, tpr, roc_auc = calculate_roc_data(model, X, y)
                
                # 绘制ROC曲线
                color = colors[base_model_type]
                line_style = line_styles[dataset_type]
                
                line, = plt.plot(fpr, tpr, 
                          color=color,
                          linestyle=line_style,
                          lw=2)
                
                # 添加到图例
                lines.append(line)
                labels.append(f"{model_name} (AUC={roc_auc:.4f})")
            else:
                print(f"模型文件 {model_file} 不存在")
        except Exception as e:
            print(f"处理模型 {model_name} 时出错: {e}")
    
    # 设置图形样式
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (False Positive Rate)', fontsize=12)
    plt.ylabel('真阳性率 (True Positive Rate)', fontsize=12)
    plt.title('各模型ROC曲线比较', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # 添加图例
    plt.legend(lines, labels, loc="lower right", fontsize=10)
    
    # 保存图形
    plt.savefig('all_models_roc_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("所有模型的ROC曲线图已保存为 all_models_roc_comparison.png")

# 在主函数中调用
if __name__ == "__main__":
    # 绘制所有模型的ROC曲线
    # plot_combined_roc()

    plot_confusion_matrix()
    
    # # 绘制所有模型的ROC曲线比较图
    # plot_all_models_roc()
    
    print("图形生成完成！")