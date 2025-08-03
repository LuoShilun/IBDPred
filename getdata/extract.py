import pandas as pd
import os

# 指定包含 MetaPhlAn 输出文件的目录
input_directory = './metaphlan_txt/'
# input_directory = './output/'

# 获取该目录下所有 .txt 文件
input_files = [f for f in os.listdir(input_directory) if f.endswith('.txt')]

# 输出文件名称
output_file = 's/species_abundance_all.csv'

# 如果文件不存在，则创建新文件
if not os.path.exists(output_file):
    # 初始化一个空字典，用于保存所有文件的数据
    all_species_data = {}

    # 遍历每个文件进行处理
    for file_name in input_files:
        file_path = os.path.join(input_directory, file_name)

        # 提取 run_accession 从文件名（假设以 'run_accession_XXXX.txt' 的格式命名）
        run_accession = file_name.split('_')[0]

        try:
            # 检查文件的行数
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if len(lines) < 6:
                    print(f"File {file_name} doesn't have enough lines (less than 6), skipping.")
                    continue  # Skip files that don't have enough lines

            # 读取 MetaPhlAn 数据文件
            data = pd.read_csv(file_path, sep='\t', comment='#', header=5)

            # 清理列名
            data.columns = data.columns.str.strip()
            data.columns = ['clade_name', 'NCBI_tax_id', 'relative_abundance', 'additional_species']

            # 过滤出物种级别的数据，去掉 't__'，只保留 's__'
            species_data = data[
                data['clade_name'].str.contains(r's__') & ~data['clade_name'].str.contains(r't__')].copy()

            # 提取物种名称
            species_data['species'] = species_data['clade_name'].str.extract(r's__([^\|]+)')

            # 仅保留 'species' 和 'relative_abundance' 列，并去除缺失数据
            species_data = species_data[['species', 'relative_abundance']].dropna(subset=['species'])

            # 将物种和相对丰度转换为字典
            species_dict = species_data.set_index('species')['relative_abundance'].to_dict()

            # 将 run_accession 作为字典的键，物种及其丰度作为数据
            if run_accession not in all_species_data:
                all_species_data[run_accession] = {}

            # 将物种及其丰度加入到字典
            all_species_data[run_accession].update(species_dict)

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    # 构建最终的 DataFrame
    # 将所有文件的数据合并为一个 DataFrame
    final_data = {'run_accession': list(all_species_data.keys())}

    # 添加每个物种的丰度
    all_species = set()
    for species_data in all_species_data.values():
        all_species.update(species_data.keys())

    # 为每个物种添加一列
    for species in all_species:
        final_data[species] = [all_species_data[run_accession].get(species, 0) for run_accession in
                               final_data['run_accession']]

    # 创建 DataFrame
    df_final = pd.DataFrame(final_data)

    # 写入到 CSV 文件
    df_final.to_csv(output_file, index=False)
    print(f"File '{output_file}' has been successfully created and data written.")

else:
    print(f"The file '{output_file}' already exists.")
