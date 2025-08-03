import pandas as pd

# 读取 Excel 文件
file_path = './NieiesnHB out.xlsx'  # 替换为您的文件路径
df = pd.read_excel(file_path)

# 提取健康程度信息并转换为数字
health_status = df.iloc[-1]  # 获取最后一行（健康程度）
health_status = health_status.replace({'healthy': 0, 'IBD': 1})  # 转换为数字

# 删除最后一行（健康程度）
df = df[:-1]

# 重置索引，因为最后一行被删除了
df = df.reset_index(drop=True)

# 转置数据框，使每一行代表一个个体
df_transposed = df.T

# 将健康信息添加到每一行的开头
df_transposed.insert(0, 'Disease', health_status)

# 保存为新的 CSV 文件
output_file_path = 'output_file.csv'  # 替换为您希望保存的文件路径
df_transposed.to_csv(output_file_path, index=False)

print("转换完成，文件已保存为:", output_file_path)