import pandas as pd

# 假设CSV文件名为'feature_importances.csv'，我们将读取这个文件
input_filename = 's/feature_importances.csv'
output_filename = 's/feature_importances_filtered.csv'

# 读取CSV文件
feature_importances = pd.read_csv(input_filename)

# 过滤掉特征重要性为0的行
filtered_importances = feature_importances[feature_importances['Importance'] != 0]

# 将过滤后的数据保存到新的CSV文件
filtered_importances.to_csv(output_filename, index=False)

print(f"Filtered feature importances have been saved to '{output_filename}'")
