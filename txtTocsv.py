import pandas as pd
import os

# --- 1. 定义文件路径和列名映射 ---
input_file_name = "00700(1).txt" # 原始输入文件，请确保它在当前目录下
output_file_name = "00700_cleaned.csv" # 清洗后的输出文件

# 定义中文到英文的列名映射字典
# 键是原始的中文列名（注意：这里使用去除空格后的中文名作为键）
column_mapping = {
    '日期': 'Date',
    '开盘': 'Open',
    '最高': 'High',
    '最低': 'Low',
    '收盘': 'Close',
    '成交量': 'Volume',
    '成交额': 'Amount'
}

encodings_to_try = ['gbk', 'gb18030', 'utf-8'] # 优先尝试GBK解决中文编码问题

# --- 2. 数据读取与处理 ---
df = None
successful_encoding = None

print(f"尝试读取文件：{input_file_name}")

for encoding in encodings_to_try:
    try:
        # 使用 pandas 读取制表符分隔的 TXT 文件
        df = pd.read_csv(
            input_file_name,
            sep='\t',
            header=0,
            skiprows=[0], # 跳过第一行（标题描述）
            encoding=encoding
        )
        successful_encoding = encoding
        break  # 读取成功，跳出循环
    except UnicodeDecodeError:
        print(f"  - 使用 {encoding} 编码失败，尝试下一种...")
    except FileNotFoundError:
        print(f"❌ 错误：未找到输入文件 '{input_file_name}'。请检查文件名是否正确。")
        exit()

# --- 3. 列名清洗、重命名与数据写入 ---
if df is not None and successful_encoding:
    
    # ① 清理原始列名：去除列名中的所有空格，以匹配映射字典的键
    original_cols = {col: col.strip() for col in df.columns}
    df.rename(columns=original_cols, inplace=True)
    
    # ② 执行列名重命名
    # 确保只重命名字典中存在的列
    df.rename(columns=column_mapping, inplace=True)
    
    # ③ 清理数据：移除所有全为空值的行（如果有的话）
    df.dropna(how='all', inplace=True)
    
    # 打印最终的列名和前几行数据进行检查
    print("-" * 40)
    print(f"✅ 文件成功读取，使用的编码是：{successful_encoding}")
    print("📢 最终列名：", df.columns.tolist())
    print("清洗后的数据前 5 行：")
    print(df.head())
    print("-" * 40)

    # 写入最终的 CSV 文件
    # index=False: 不写入行索引
    df.to_csv(output_file_name, index=False, encoding='utf-8')

    print(f"✅ 转换和重命名成功！")
    print(f"新的 CSV 文件已保存到：{output_file_name}")
    
else:
    print("❌ 转换失败：所有尝试的编码都无法正确解析文件。")