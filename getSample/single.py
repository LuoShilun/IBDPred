from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def fetch_sample_health_info(sample_url):
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

    try:
        driver.get(sample_url)

        # 等待并点击隐私政策同意按钮
        try:
            print("等待隐私政策同意按钮...")
            WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "/html/body/div[7]/div/div/button"))
            ).click()
            print("已同意隐私政策")
        except Exception as e:
            print(f"未找到隐私政策同意按钮: {e}")

        # 等待表格加载
        print("等待表格加载...")
        try:
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="view-content-col"]/div[4]/div/div[6]/app-thirdparty-curations/div/section/div/table'))
            )
            print("表格已加载")
        except Exception as e:
            print(f"表格加载超时或未找到: {e}")
            print("当前页面内容：")
            print(driver.page_source)  # 打印当前页面的HTML内容
            return None

        # 尝试获取所需信息
        try:
            table = driver.find_element(By.XPATH, '//*[@id="view-content-col"]/div[4]/div/div[6]/app-thirdparty-curations/div/section/div/table')
            rows = table.find_elements(By.TAG_NAME, 'tr')[1:]  # 跳过表头行

            # 初始化信息字典
            sample_data = {
                'Sample Accession': 'SAMEA2042646',
                'Age': 'N/A',
                'Health State': 'N/A',
                'BMI': 'N/A',
                'Lifestyle': 'N/A'
            }

            for row in rows:
                cols = row.find_elements(By.TAG_NAME, 'td')
                if len(cols) >= 3:  # 确保有足够的列
                    key = cols[1].text.strip()
                    value = cols[2].text.strip()

                    # 更新字典中的相应健康属性
                    if "Age" == key:
                        sample_data['Age'] = value
                    elif "Health state" in key:
                        sample_data['Health State'] = value
                    elif "BMI" in key:
                        sample_data['BMI'] = value
                    elif "Lifestyle" in key:
                        sample_data['Lifestyle'] = value

            return sample_data

        except Exception as e:
            print(f"抓取数据时出现错误: {e}")
            return None

    except Exception as e:
        print(f"出现错误: {e}")
        return None

    finally:
        driver.quit()

# 示例运行
if __name__ == "__main__":
    sample_url = "https://www.ebi.ac.uk/ena/browser/view/SAMEA2042646?show=curations"
    sample_data = fetch_sample_health_info(sample_url)
    print(sample_data)