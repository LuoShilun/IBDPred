# import csv
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service as ChromeService
# from webdriver_manager.chrome import ChromeDriverManager
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# import time
#
#
# def fetch_samples_health_info(project_accession):
#     driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
#
#     # Open CSV file once for writing
#     with open('in.csv', mode='w', newline='', encoding='utf-8') as csv_file:
#         fieldnames = ['Run Accession', 'Sample Accession']
#         writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#         writer.writeheader()  # Write the header once
#
#         try:
#             base_url = f"https://www.ebi.ac.uk/ena/browser/view/{project_accession}"
#             driver.get(base_url)
#
#             # 等待并点击隐私政策同意按钮,手动点击
#             time.sleep(10)
#
#             while True:
#                 # 等待表格加载
#                 WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, 'table')))
#                 table = driver.find_element(By.TAG_NAME, 'table')
#                 rows = table.find_elements(By.TAG_NAME, 'tr')[1:]  # 跳过表头行
#
#                 samples_info = []  # This is now inside the loop to store data for each page
#                 for row in rows:
#                     cols = row.find_elements(By.TAG_NAME, 'td')
#                     if cols:
#                         run_accession = cols[3].text.strip()  # Run Accession在第3列
#                         sample_accession = cols[1].text.strip()  # Sample Accession在第1列
#
#                         samples_info.append({
#                             'Run Accession': run_accession,
#                             'Sample Accession': sample_accession
#                         })
#
#                 # Write the page's data to CSV after processing
#                 for sample in samples_info:
#                     writer.writerow(sample)
#
#                 # 查找并点击“下一页”按钮
#                 try:
#                     # 使用显式等待确保按钮可点击
#                     next_button = WebDriverWait(driver, 5).until(
#                         EC.element_to_be_clickable((By.XPATH,
#                                                     "//*[@id='view-content-col']/div[4]/div/div[2]/app-read-file-links/mat-paginator/div/div/div[2]/button[3]"))
#                     )
#                     next_button.click()
#                     time.sleep(1)  # 等待页面加载
#                 except Exception as e:
#                     print("未找到下一页按钮或无法点击，结束分页.")
#                     break
#
#         except Exception as e:
#             print(f"出现错误: {e}")
#
#         finally:
#             driver.quit()
#
#     print("样本信息已导出到 in.csv")
#
#
# # 示例运行
# if __name__ == "__main__":
#     project_accession = "PRJEB1220"  # 替换为实际项目编号
#     fetch_samples_health_info(project_accession)


import csv

# 定义输入和输出文件名
health_filename = 'health.csv'
samples_filename = './samples_transfer/samples.csv'
output_filename = 'samples_health.csv'

# 读取health.csv文件并创建一个字典，以run_accession和sample_accession为键，其他信息为值
health_data = {}
with open(health_filename, 'r', encoding='utf-8') as health_file:
    health_reader = csv.DictReader(health_file)
    for row in health_reader:
        key = (row['Run Accession'], row['Sample Accession'])
        health_data[key] = {
            'Age': row['Age'],
            'BMI': row['BMI'],
            'Health State': row['Health State'],
            'Disease Name': row['Disease Name'],
            'Lifestyle': row['Lifestyle']
        }

# 读取samples.csv文件，并根据run_accession和sample_accession匹配health.csv中的数据
with open(samples_filename, 'r', encoding='utf-8') as samples_file, \
        open(output_filename, 'w', newline='', encoding='utf-8') as output_file:
    samples_reader = csv.reader(samples_file)
    output_writer = csv.writer(output_file)

    # 写入output.csv文件的表头
    output_writer.writerow(
        ['run_accession', 'sample_accession', 'Age', 'BMI', 'Health State', 'Disease Name', 'Lifestyle'])

    for row in samples_reader:
        run_accession, sample_accession = row
        key = (run_accession, sample_accession)

        if key in health_data:
            # 如果在health_data中找到了匹配的键，写入对应的健康信息
            health_info = health_data[key]
            output_writer.writerow([
                run_accession,
                sample_accession,
                health_info.get('Age', ''),
                health_info.get('BMI', ''),
                health_info.get('Health State', ''),
                health_info.get('Disease Name', ''),
                health_info.get('Lifestyle', '')
            ])
        else:
            # 如果没有找到匹配的键，写入空值
            output_writer.writerow([
                run_accession,
                sample_accession,
                '', '', '', '', ''
            ])

print("out.csv文件已生成。")