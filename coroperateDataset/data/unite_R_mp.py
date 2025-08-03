# import pandas as pd
#
# # 读取原有数据集
# original_file_path = './data/s/data2.csv'  # 替换为原数据集文件的路径
# new_file_path = './data/R_data/r_data.csv'  # 替换为新数据集文件的路径
#
# original_df = pd.read_csv(original_file_path)
# new_df = pd.read_csv(new_file_path)
#
# # 确保第一列是健康状态，并将其设置为索引
# original_df.set_index('Disease Name', inplace=True)
# new_df.set_index('Disease Name', inplace=True)
#
# # 合并数据集，以原有数据集为基准
# merged_df = original_df.join(new_df, how='outer', rsuffix='_new')
#
# # 用0填充缺失值
# merged_df.fillna(0, inplace=True)
#
# # 将合并后的数据集保存为新的CSV文件
# output_file_path = './data/dataset.csv'  # 替换为希望保存的文件路径
# merged_df.to_csv(output_file_path)
#
# print(f"数据合并完成，已保存至：{output_file_path}")

import pandas as pd

# 读取CSV文件，指定分隔符为任意数量的空格
df = pd.read_csv('R_data/r_data.csv', sep=r'\s+')

# 将所有特征名中的空格替换为下划线
df.columns = [col.replace(' ', '_') for col in df.columns]

# 导出修改后的CSV文件
df.to_csv('./data/R_data/r_data.csv', index=False)

print("CSV文件已成功修改并导出为 'modified_dataset.csv'")