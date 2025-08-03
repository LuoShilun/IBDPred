import csv

# 定义输入和输出文件名
input_filename = 'filereport_read_run_PRJEB1220_tsv (1).txt'
output_filename = 'samples.csv'

# 打开输入文件和输出文件
with open(input_filename, 'r', encoding='utf-8') as infile, \
     open(output_filename, 'w', newline='', encoding='utf-8') as outfile:

    # 创建CSV读取器
    reader = csv.reader(infile, delimiter='\t')

    # 创建CSV写入器
    writer = csv.writer(outfile)

    # 写入CSV文件的表头
    writer.writerow(['run_accession', 'sample_accession'])

    # 逐行读取数据并写入CSV文件
    for row in reader:
        # 只提取run_accession和sample_accession字段
        writer.writerow([row[0], row[1]])

print("CSV文件已生成。")