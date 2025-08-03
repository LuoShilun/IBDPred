import csv
import pandas as pd

data = pd.read_csv("./samples_transfer/samples.csv")
#获取某一列的长度（去重）
# print(data.groupby('sample_accession').size())
# print(data.groupby('run_accession').size())
#
#描述性统计
print(data.describe())

# data = data.drop(['Sample URL'],axis=1)
# data.to_csv("health.csv",index=False)


# import json
# import csv
#
# filename = 'data.txt'
#
# # 读取JSON数据
# with open(filename, 'r', encoding='utf-8') as file:
#     # 由于文件可能包含多个JSON对象，我们将其视为一个JSON数组
#     data = json.load(file)
#
# # 打开一个新的CSV文件用于写入
# with open('ijauz.csv', 'w', newline='', encoding='utf-8') as csvfile:
#     # 创建一个csv.DictWriter对象
#     fieldnames = data[0].keys()  # 假设所有字典都有相同的键
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#
#     # 写入表头
#     writer.writeheader()
#
#     # 写入数据
#     for item in data:
#         writer.writerow(item)
#
# print("CSV文件已生成。")