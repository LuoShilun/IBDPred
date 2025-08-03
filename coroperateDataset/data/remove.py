import pandas as pd

# 读取 data1.csv 文件
data_file = 's/data1.csv'
data = pd.read_csv(data_file)

# 去除特定的列
columns_to_remove = ['run_accession', 'sample_accession', 'Age', 'BMI', 'Health State', 'Lifestyle']
data = data.drop(columns=columns_to_remove)

# 更新 Health State 列的值
data['Disease Name'] = data['Disease Name'].apply(lambda x: 0 if pd.isna(x) or x == 'Obesity' else (1 if x in ['UC', 'CD', 'IBD'] else x))

# 保存处理后的数据到新的 CSV 文件
output_file = 's/data2.csv'
data.to_csv(output_file, index=False)

print(f"Data has been processed and saved to '{output_file}'")
