import pandas as pd

# 输入文件路径
file1 = 'prjeb1220_dataset.csv'  # 第一个文件
file2 = 'hmp2019.csv'  # 第二个文件

# 输出文件路径
output_file1 = 'prjeb1220_common_columns.csv'  # 修改后的第一个文件
output_file2 = 'hmp2019_dataset_common_columns.csv'  # 修改后的第二个文件

# 读取两个文件
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# 获取两个文件的列名
columns1 = set(df1.columns)
columns2 = set(df2.columns)

# 找到两个文件中公有的列名
common_columns = list(columns1.intersection(columns2))

# 保留公有的列名，删除其他列
df1_common = df1[common_columns]
df2_common = df2[common_columns]

# 保存修改后的文件
df1_common.to_csv(output_file1, index=False, encoding='utf-8')
df2_common.to_csv(output_file2, index=False, encoding='utf-8')

print(f"文件 {file1} 已修改并保存为 {output_file1}")
print(f"文件 {file2} 已修改并保存为 {output_file2}")