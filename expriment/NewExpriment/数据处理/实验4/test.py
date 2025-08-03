import pymysql
from pymysql import OperationalError
from typing import List, Dict, Any, Optional


class DatabaseTableExtractor:
    """数据库表结构提取工具，支持连接多个数据库并获取指定表的定义"""

    def __init__(self, db_configs: List[Dict[str, Any]]):
        self.db_configs = db_configs

    def get_table_definitions(self, db_alias: str, table_names: List[str]) -> Dict[str, str]:
        """获取指定数据库中多个表的定义"""
        db_config = next((cfg for cfg in self.db_configs if cfg.get('alias') == db_alias), None)
        if not db_config:
            raise ValueError(f"未找到别名 '{db_alias}' 对应的数据库配置")

        results = {}
        try:
            conn = pymysql.connect(
                host=db_config['host'],
                port=db_config.get('port', 3306),
                user=db_config['user'],
                password=db_config['password'],
                database=db_config['database'],
                charset=db_config.get('charset', 'utf8mb4'),
                cursorclass=pymysql.cursors.DictCursor
            )

            with conn.cursor() as cursor:
                for table in table_names:
                    try:
                        cursor.execute(f"SHOW CREATE TABLE `{table}`")
                        result = cursor.fetchone()
                        if result:
                            results[table] = result['Create Table']
                        else:
                            results[table] = f"表 '{table}' 不存在"
                    except Exception as e:
                        results[table] = f"获取表定义失败: {str(e)}"
        except OperationalError as e:
            results["error"] = f"数据库连接失败: {str(e)}"
        finally:
            if conn:
                conn.close()

        return results


def main(
        db_configs: List[Dict[str, Any]],
        db_table_map: Dict[str, List[str]]
) -> Dict[str, Dict[str, str]]:

    extractor = DatabaseTableExtractor(db_configs)
    all_results = {}

    for db_alias, tables in db_table_map.items():
        try:
            all_results[db_alias] = extractor.get_table_definitions(db_alias, tables)
        except Exception as e:
            all_results[db_alias] = {"error": f"处理数据库 '{db_alias}' 时出错: {str(e)}"}

    return {
        "table_definitions": all_results
    }


# db_configs = [
#         {
#             "alias": "sports",  # 数据库别名，用于标识不同数据库
#             "host": "171.35.137.77",
#             "port": 3312,
#             "user": "root",
#             "password": "jxlt@lic292",
#             "database": "sports"
#         }
#     ]
# db_table_requests = {
#     "sports": ["t_field", "t_game"],  # 从db1获取users和orders表
# }