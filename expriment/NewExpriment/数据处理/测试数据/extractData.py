import pandas as pd

# # 1. 读入原始数据
# df = pd.read_csv('./hab.csv', index_col=0)  # 第一列是特征名，设为行索引
#
# # # 2. 转置：行列互换
# # df = df.T
# #
# #  #规范化列名：空格 → 下划线
# # df.columns = [col.strip().replace(' ', '_') for col in df.columns]
#
# df['Diease'] = df['Diease'].map({'IBD': 1, 'healthy': 0})  # 替换
# # 3. 保存结果
# df.to_csv('hab.csv', index=True)   # 第一列仍是样本名


#提取共同特征
import pandas as pd
from pathlib import Path

# 1. 设置路径
train_path = 'r_dataset.csv'  # 替换为你的训练集文件名
test_path  = '../../原始数据/hab.csv'  # 替换为你的测试集文件名
out_dir    = '../../实验4'  # 输出目录，可改成任意路径

# 2. 读入数据
df_train = pd.read_csv(train_path)
df_test  = pd.read_csv(test_path)

# 3. 找共同列（去掉样本名列，如果有的话）
#    假设样本名放在第一列，名字叫 Sample_id / SampleID / 其他
id_col = None
for c in df_train.columns:
    if c.lower() in {'sample_id', 'sampleid', 'sample'}:
        id_col = c
        break

if id_col:               # 如果存在样本名列，把它排除在特征外
    feats_train = set(df_train.columns) - {id_col}
    feats_test  = set(df_test.columns)  - {id_col}
else:                    # 否则直接用全部列
    feats_train = set(df_train.columns)
    feats_test  = set(df_test.columns)

common_feats = sorted(feats_train & feats_test)   # 排序保证顺序一致

if id_col:               # 把样本名列加回来
    common_cols = [id_col] + common_feats
else:
    common_cols = common_feats

# 4. 提取并保存
df_train_common = df_train[common_cols]
df_test_common  = df_test[common_cols]

df_train_common.to_csv(Path(out_dir) / 'r.csv', index=False)
df_test_common.to_csv(Path(out_dir) / 'hab.csv', index=False)

print(f'共同特征 {len(common_feats)} 个，已保存：')
print(common_feats)

# #数据集分析
# import pandas as pd
# df = pd.read_csv('prjeb1220.csv')          # 1. 读文件
# print('样本数:', len(df))                  # 2. 样本总量
# print(df['Disease'].value_counts())        # 3. 目标分布（IBD / healthy 计数）
# print(df.isnull().sum().sum(), '个缺失值') # 4. 缺失概览

# #提取ijauz和hab共同特征，作为新的测试集
# import pandas as pd
#
# # 加载两个数据集
# ijauz_df = pd.read_csv('ijauz.csv', header=0)
# hab_df = pd.read_csv('hab.csv', header=0)
#
# # 获取两个数据集的共同特征（列名）
# common_features = list(set(ijauz_df.columns) & set(hab_df.columns))
#
# # 提取共同特征组成新的数据集
# new_ijauz_df = ijauz_df[common_features]
# new_hab_df = hab_df[common_features]
#
# # 合并两个新的数据集为一个数据集
# combined_df = pd.concat([new_ijauz_df, new_hab_df], axis=0, ignore_index=True)
#
# # 将合并后的数据集保存为CSV文件
# combined_df.to_csv('combined_dataset.csv', index=False)
#
# print("合并后的数据集已保存为 combined_dataset.csv。")



# import pandas as pd
#
# # 加载数据
# file_path = 'prjeb1220.csv'
# data = pd.read_csv(file_path)
#
# # 检查是否有足够的样本
# disease_0_count = (data['Disease'] == 0).sum()
# disease_1_count = (data['Disease'] == 1).sum()
#
# if disease_0_count < 441 or disease_1_count < 441:
#     raise ValueError("数据集中没有足够的样本满足要求。")
#
# # 随机抽取441个Disease为0的样本，不重复
# disease_0_samples = data[data['Disease'] == 0].sample(n=441, replace=False, random_state=42)
#
# # 随机抽取441个Disease为1的样本，不重复
# disease_1_samples = data[data['Disease'] == 1].sample(n=441, replace=False, random_state=42)
#
# # 合并两个样本集
# new_dataset = pd.concat([disease_0_samples, disease_1_samples])
#
# # 保存到新的CSV文件
# new_dataset.to_csv('new_prjeb1220.csv', index=False)
#
# print("新数据集已成功保存到 new_dataset.csv 文件中。")