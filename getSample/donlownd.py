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
# # 定义一个函数来抓取单个样本的健康信息
# def fetch_sample_health_info(sample_url):
#     driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
#     sample_data = {'Age': 'N/A', 'BMI': 'N/A', 'Health State': 'N/A', 'Disease Name': 'N/A', 'Lifestyle': 'N/A'}
#
#     try:
#         driver.get(sample_url)
#
#         # 等待并点击隐私政策同意按钮（如有需要）
#         try:
#             WebDriverWait(driver, 1).until(
#                 EC.element_to_be_clickable((By.XPATH, "/html/body/div[7]/div/div/button"))
#             ).click()
#         except Exception as e:
#             print("隐私政策按钮未找到或无需点击")
#
#         # 等待页面表格加载
#         WebDriverWait(driver, 3).until(
#             EC.presence_of_element_located((By.XPATH,
#                                             '//*[@id="view-content-col"]/div[4]/div/div[6]/app-thirdparty-curations/div/section/div/table'))
#         )
#
#         # 抓取表格中的健康信息
#         table = driver.find_element(By.XPATH,
#                                     '//*[@id="view-content-col"]/div[4]/div/div[6]/app-thirdparty-curations/div/section/div/table')
#         rows = table.find_elements(By.TAG_NAME, 'tr')[1:]  # 跳过表头行
#
#         for row in rows:
#             cols = row.find_elements(By.TAG_NAME, 'td')
#             if len(cols) >= 3:
#                 key = cols[1].text.strip()
#                 value = cols[2].text.strip()
#
#                 if "Age" == key:
#                     sample_data['Age'] = value
#                 elif "Health state" in key:
#                     sample_data['Health State'] = value
#                 elif "BMI" in key:
#                     sample_data['BMI'] = value
#                 elif "Lifestyle" in key:
#                     sample_data['Lifestyle'] = value
#
#         # 如果健康状态是Healthy，则疾病名称为N/A
#         if sample_data['Health State'] != 'Healthy':
#             # 尝试获取“Disease name”信息
#             for row in rows:
#                 cols = row.find_elements(By.TAG_NAME, 'td')
#                 if len(cols) >= 3 and "Disease name" in cols[1].text:
#                     sample_data['Disease Name'] = cols[2].text.strip()
#
#     except Exception as e:
#         print(f"抓取数据时出现错误: {e}")
#
#     finally:
#         driver.quit()
#
#     return sample_data
#
#
# # 定义一个函数来读取CSV文件，并为每个样本抓取健康信息
# def process_samples_info(csv_input_file, csv_output_file):
#     with open(csv_input_file, mode='r', newline='', encoding='utf-8') as infile:
#         reader = csv.DictReader(infile)
#         samples_info = list(reader)  # 读取所有记录
#
#     # 定义字段名并打开输出CSV文件
#     fieldnames = ['Run Accession', 'Sample Accession','Sample URL', 'Age', 'BMI', 'Health State', 'Disease Name', 'Lifestyle']
#     with open(csv_output_file, mode='w', newline='', encoding='utf-8') as outfile:
#         writer = csv.DictWriter(outfile, fieldnames=fieldnames)
#         writer.writeheader()
#         number = 1
#         # 逐个样本抓取健康信息并写入新文件
#         for sample in samples_info:
#             print(f"处理样本[{number}]: {sample['Sample Accession']} ({sample['Sample URL']})")
#             sample_url = sample['Sample URL']
#             health_data = fetch_sample_health_info(sample_url)
#
#             # 将抓取的健康信息与原有的Run Accession和Sample Accession结合
#             sample.update(health_data)
#             writer.writerow(sample)  # 写入新CSV文件
#             number = number +1
#
#     print(f"样本健康信息已导出到 {csv_output_file}")
#
#
# # 示例运行
# if __name__ == "__main__":
#     input_csv = "in.csv"  # 输入CSV文件
#     output_csv = "samples_health_info.csv"  # 输出CSV文件
#     process_samples_info(input_csv, output_csv)
#

#多线程快捷版
import csv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import time



def fetch_sample_health_info(sample_url, retries=3):
    sample_data = {'Age': 'N/A', 'BMI': 'N/A', 'Health State': 'N/A', 'Disease Name': 'N/A', 'Lifestyle': 'N/A'}

    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

    for attempt in range(retries):
        try:
            driver.get(sample_url)

            # 等待并点击隐私政策同意按钮（如有需要）
            try:
                WebDriverWait(driver, 1).until(
                    EC.element_to_be_clickable((By.XPATH, "/html/body/div[7]/div/div/button"))
                ).click()
            except Exception:
                print("隐私政策按钮未找到或无需点击")

            # 等待页面表格加载
            WebDriverWait(driver, 3).until(
                EC.presence_of_element_located((By.XPATH,
                                                '//*[@id="view-content-col"]/div[4]/div/div[6]/app-thirdparty-curations/div/section/div/table'))
            )

            # 抓取表格中的健康信息
            table = driver.find_element(By.XPATH,
                                        '//*[@id="view-content-col"]/div[4]/div/div[6]/app-thirdparty-curations/div/section/div/table')
            rows = table.find_elements(By.TAG_NAME, 'tr')[1:]  # 跳过表头行

            for row in rows:
                cols = row.find_elements(By.TAG_NAME, 'td')
                if len(cols) >= 3:
                    key = cols[1].text.strip()
                    value = cols[2].text.strip()

                    if "Age" == key:
                        sample_data['Age'] = value
                    elif "Health state" in key:
                        sample_data['Health State'] = value
                    elif "BMI" in key:
                        sample_data['BMI'] = value
                    elif "Lifestyle" in key:
                        sample_data['Lifestyle'] = value

            # 如果健康状态是Healthy，则疾病名称为N/A
            if sample_data['Health State'] != 'Healthy':
                # 尝试获取“Disease name”信息
                for row in rows:
                    cols = row.find_elements(By.TAG_NAME, 'td')
                    if len(cols) >= 3 and "Disease name" in cols[1].text:
                        sample_data['Disease Name'] = cols[2].text.strip()

            return sample_data

        except Exception as e:
            print(f"抓取数据时出现错误: {e}")
            traceback.print_exc()  # 打印完整的异常堆栈
            time.sleep(2)  # 等待2秒后重试


    return sample_data


# 定义一个函数来读取CSV文件，并为每个样本抓取健康信息
def process_samples_info(csv_input_file, csv_output_file):
    with open(csv_input_file, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        samples_info = list(reader)  # 读取所有记录

    # 定义字段名并打开输出CSV文件
    fieldnames = ['Run Accession', 'Sample Accession', 'Sample URL', 'Age', 'BMI', 'Health State', 'Disease Name',
                  'Lifestyle']

    with open(csv_output_file, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        # 使用 ThreadPoolExecutor 进行并发抓取
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_sample = {executor.submit(fetch_sample_health_info, sample['Sample URL']): sample for
                                sample in samples_info}
            number = 0
            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]
                try:
                    number = number+1
                    health_data = future.result()
                    sample.update(health_data)  # 将抓取的健康信息与原有的样本信息结合
                    writer.writerow(sample)  # 写入新CSV文件
                    print(f"处理样本[{number}]: {sample['Sample Accession']} 完成")
                except Exception as e:
                    print(f"处理样本[{number}] {sample['Sample Accession']} 时出现错误: {e}")

    print(f"样本健康信息已导出到 {csv_output_file}")


# 示例运行
if __name__ == "__main__":
    input_csv = "samples_info.csv"  # 输入CSV文件
    output_csv = "samples_health_info.csv"  # 输出CSV文件
    process_samples_info(input_csv, output_csv)
