import csv

# 定义输入和输出文件名
input_filename = 'data/samples_health.csv'
output_filename = 'out_filled.csv'

# 读取out.csv文件并创建一个字典，以(run_accession, sample_accession)为键，其他信息为值
combined_data = {}
with open(input_filename, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        key = (row['run_accession'], row['sample_accession'])
        if key not in combined_data:
            combined_data[key] = {k: v for k, v in row.items() if k in ['Age', 'BMI', 'Health State', 'Disease Name', 'Lifestyle']}
        else:
            for key in ['Age', 'BMI', 'Health State', 'Disease Name', 'Lifestyle']:
                if not combined_data[key].get(key) and row[key]:
                    combined_data[key][key] = row[key]

# 将处理后的数据写入out_filled.csv文件
with open(output_filename, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # 写入out_filled.csv文件的表头
    writer.writerow(['run_accession', 'sample_accession', 'Age', 'BMI', 'Health State', 'Disease Name', 'Lifestyle'])
    # 写入数据
    for key, values in combined_data.items():
        run_accession, sample_accession = key
        writer.writerow([
            run_accession,
            sample_accession,
            values.get('Age', ''),
            values.get('BMI', ''),
            values.get('Health State', ''),
            values.get('Disease Name', ''),
            values.get('Lifestyle', '')
        ])

print("out_filled.csv文件已生成。")