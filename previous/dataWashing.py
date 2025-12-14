import pandas as pd
import numpy as np

# --- 1. 定义文件和划分时间 ---
cleaned_file_name = "00700_cleaned.csv"
TRAIN_END_DATE = '2023-12-31'
INVEST_START_DATE = '2024-01-01'

try:
    # 读取已转换格式的 CSV 文件
    df = pd.read_csv(cleaned_file_name)

    # 步骤一：数据结构检查与格式转换
    # ----------------------------------------------------
    # 1. 转换日期格式并设置为索引
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # 2. 确保价格和成交量/额是数值类型
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount']
    # 使用 errors='coerce' 将非数字值转换为 NaN
    df[price_cols] = df[price_cols].apply(pd.to_numeric, errors='coerce')

    print("数据读取和类型转换完成。")
    print(df.dtypes)
    print("-" * 40)


    # 步骤二：缺失值和异常值处理
    # ----------------------------------------------------
    # 1. 检查缺失值
    print("缺失值检查：")
    print(df.isnull().sum())
    
    # 2. 填充缺失值 (如果存在)
    # 采用前向填充，假设数据点缺失是由于数据记录问题，价格应保持前一天的状态
    if df.isnull().values.any():
        df.fillna(method='ffill', inplace=True)
        print("\n已使用前向填充 (ffill) 处理缺失值。")

    # 3. 处理异常值：移除成交量为 0 的行 (通常不是正常的交易日)
    initial_rows = len(df)
    df = df[df['Volume'] > 0]
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        print(f"已移除 {removed_rows} 行成交量为零的记录。")
    print("-" * 40)
    
    
    # 步骤三：时间序列连续性检查与数据集划分
    # ----------------------------------------------------
    
    # 1. 严格划分训练集和测试/投资集
    # loc[:'2023-12-31'] 包含截止日期
    train_df = df.loc[df.index <= TRAIN_END_DATE].copy()
    # loc['2024-01-01':] 包含起始日期
    test_df = df.loc[df.index >= INVEST_START_DATE].copy() 
    
    # 2. 检查划分是否正确
    print(f"训练集日期范围：{train_df.index.min()} 至 {train_df.index.max()}")
    print(f"投资集日期范围：{test_df.index.min()} 至 {test_df.index.max()}")
    print(f"训练集大小：{len(train_df)} 个交易日")
    print(f"投资集大小：{len(test_df)} 个交易日")

    # 提示下一步：因子挖掘
    print("-" * 40)
    print("✅ 数据清洗和数据集划分完成。")
    print("👉 下一步：开始进行因子挖掘（特征工程）和构建标签（Y）。")

except FileNotFoundError:
    print(f"❌ 错误：未找到文件 '{cleaned_file_name}'。请确保您已经运行了上一步的代码并成功生成了该文件。")
except Exception as e:
    print(f"❌ 发生错误：{e}")