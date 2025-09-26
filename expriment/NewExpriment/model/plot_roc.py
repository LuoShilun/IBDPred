import pickle
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示支持
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def load_cv_results(filename):
    """加载保存的交叉验证结果"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def plot_ensemble_models_from_saved_data():
    """从保存的数据文件中绘制所有集成模型的ROC曲线"""
    plt.figure(figsize=(12, 8))
    
    # 定义颜色和线型
    colors = {
        'average': '#2196F3',  # 蓝色
        'voting': '#4CAF50',   # 绿色
        'stacking': '#FFC107', # 黄色
        'improved_stacking': '#F44336'  # 红色
    }
    
    line_styles = {
        'rf': '-',   # 实线
        'mi': '--'   # 虚线
    }
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7)
    
    # 尝试加载各个模型的结果
    try:
        # 加权平均模型
        average_results = load_cv_results('ModelPKL/average_cv_results.pkl')
        for method, result in average_results.items():
            plt.plot(result['mean_fpr'], result['mean_tpr'],
                    color=colors['average'],
                    label=f'Weighted Average-{method.upper()}(AUC={result["auc"]:.4f})',
                    lw=2.5, linestyle=line_styles[method])
    except:
        print("未找到加权平均模型的结果文件")
    
    try:
        # 软投票模型
        voting_results = load_cv_results('ModelPKL/voting_cv_results.pkl')
        for method, result in voting_results.items():
            plt.plot(result['mean_fpr'], result['mean_tpr'],
                    color=colors['voting'],
                    label=f'Soft Voting-{method.upper()}(AUC={result["auc"]:.4f})',
                    lw=2.5, linestyle=line_styles[method])
    except:
        print("未找到软投票模型的结果文件")
    
    try:
        # 标准Stacking模型
        stacking_results = load_cv_results('ModelPKL/stacking_cv_results.pkl')
        for method, result in stacking_results.items():
            plt.plot(result['mean_fpr'], result['mean_tpr'],
                    color=colors['stacking'],
                    label=f'Stacking-{method.upper()}(AUC={result["auc"]:.4f})',
                    lw=2.5, linestyle=line_styles[method])
    except:
        print("未找到标准Stacking模型的结果文件")
    
    try:
        # 改进Stacking模型
        improved_stacking_results = load_cv_results('ModelPKL/stacking_improved_cv_results.pkl')
        for method, result in improved_stacking_results.items():
            plt.plot(result['mean_fpr'], result['mean_tpr'],
                    color=colors['improved_stacking'],
                    label=f'Improved Stacking-{method.upper()}(AUC={result["auc"]:.4f})',
                    lw=2.5, linestyle=line_styles[method])
    except:
        print("未找到改进Stacking模型的结果文件")
    
    # 设置图形样式
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title('ROC Curves of the Improved Stacking Model on Different Datasets', fontsize=20)
    plt.grid(True, alpha=0.3)
    
    # 添加图例
    plt.legend(loc="lower right", fontsize=16)
    
    # 保存图形
    plt.savefig('./resultsImage/ensemble_models_roc_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("所有集成模型的ROC曲线图已保存为 ensemble_models_roc_comparison.png")

if __name__ == "__main__":
    plot_ensemble_models_from_saved_data()