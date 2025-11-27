import pandas as pd
import numpy as np
import warnings
from collections import Counter

# 忽略 pandas 的 SettingWithCopyWarning
warnings.filterwarnings("ignore")

# --- 1. 配置参数 (严格遵守项目要求) ---
INPUT_FILE_NAME = "00700(1).txt"
CLASSIFICATION_THRESHOLD = 0.005  # 涨跌阈值 delta = 0.5% 

# 项目定义的关键日期
TRAIN_END_DATE = '2023-12-31'
INVEST_START_DATE = '2024-01-01'
INVEST_END_DATE = '2025-04-24' # 投资截止日期

# 最终选择的 Top 5 因子 (基于LightGBM评估结果)
FINAL_FEATURE_SET = [
    'Return_Lag_1', 'Return_Lag_5', 'Return_Lag_2', 
    'Daily_Return', 'Body_Ratio'
]
FINAL_COLUMNS = FINAL_FEATURE_SET + ['Target'] 

# 输出文件名
TRAIN_FILE_NAME = "00700_train_data_final.csv"
PREDICTING_FILE_NAME = "00700_predicting_data_final.csv"

# 中文到英文的列名映射字典
COLUMN_MAPPING = {
    '日期': 'Date', '开盘': 'Open', '最高': 'High', '最低': 'Low', 
    '收盘': 'Close', '成交量': 'Volume', '成交额': 'Amount'
}
encodings_to_try = ['gbk', 'gb18030', 'utf-8'] 

# --- 2. 文件读取和格式转换 ---

def load_and_preprocess_raw_data(file_name):
    """读取原始 TXT 文件并转换为 DataFrame。"""
    df = None
    print(f"尝试读取文件：{file_name}")
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_name, sep='\t', header=0, skiprows=[0], encoding=encoding)
            print(f"  - 成功使用 {encoding} 编码读取。")
            break
        except Exception:
            continue

    if df is None:
        print("❌ 读取失败：所有尝试的编码都无法正确解析文件。")
        return None

    # 清理和重命名
    original_cols = {col: col.strip() for col in df.columns}
    df.rename(columns=original_cols, inplace=True)
    df.rename(columns=COLUMN_MAPPING, inplace=True)
    
    # 类型转换
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount']
    df[price_cols] = df[price_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(how='all', inplace=True)
    
    return df.copy()

# --- 3. 完整的特征工程函数 ---

def feature_engineering(df):
    """计算所有 Top 5 因子所需的特征，以及 Target 标签。"""
    
    # 基础清洗与标签构建 (Y)
    df.fillna(method='ffill', inplace=True)
    df = df[df['Volume'] > 0].copy()
    
    # 1. 目标标签 Target (Y)
    df['Next_Day_Close'] = df['Close'].shift(-1)
    df['Next_Day_Return'] = (df['Next_Day_Close'] / df['Close']) - 1
    df['Target'] = 0  
    df.loc[df['Next_Day_Return'] > CLASSIFICATION_THRESHOLD, 'Target'] = 1
    df.loc[df['Next_Day_Return'] < -CLASSIFICATION_THRESHOLD, 'Target'] = -1
    
    # 2. Daily_Return (R_t) 和滞后特征
    df['Daily_Return'] = df['Close'].pct_change() 
    LAG_N = 5
    for i in range(1, LAG_N + 1):
        df[f'Return_Lag_{i}'] = df['Daily_Return'].shift(i)

    # 3. Body_Ratio
    df['True_Range'] = df['High'] - df['Low']
    df['Body_Length'] = abs(df['Close'] - df['Open'])
    # Body_Ratio: 避免除以零
    df['Body_Ratio'] = (df['Body_Length'] / df['True_Range']).replace(np.inf, 1).fillna(0) 

    # 清理所有 NaN 值（大部分由滞后窗口和 Body_Ratio 导致）
    df.dropna(inplace=True)

    return df

# --- 4. 主程序运行：特征提取、分割和保存 ---

if __name__ == "__main__":
    
    # 1. 读取和计算所有特征
    df_full = load_and_preprocess_raw_data(INPUT_FILE_NAME)
    
    if df_full is not None:
        df_with_features = feature_engineering(df_full)
        
        # 2. 筛选最终特征集 (Top 5 因子 + Target)
        # 检查是否所有列都存在
        missing_cols = [col for col in FINAL_COLUMNS if col not in df_with_features.columns]
        if missing_cols:
            print(f"❌ 错误：在计算出的特征中缺少以下列：{missing_cols}。请检查 feature_engineering 函数。")
            exit()
            
        df_final = df_with_features[FINAL_COLUMNS].copy()
        
        print("-" * 50)
        print(f"✅ 最终特征集筛选完成！总有效样本数: {len(df_final)}。")
        print(f"选取的特征和标签：{FINAL_COLUMNS}")

        # 3. 按时间点分割数据集 (严格的非重叠时间划分)
        
        # 训练集: 2018/01/02 到 2023/12/31
        df_train = df_final.loc[df_final.index <= TRAIN_END_DATE].copy()
        
        # 投资集/预测集: 2024/01/01 到 2025/04/24
        # 使用 loc[start:end] 确保只包含所需时间段
        df_predicting = df_final.loc[(df_final.index >= INVEST_START_DATE) & (df_final.index <= INVEST_END_DATE)].copy()

        # 4. 保存为 CSV 文件
        
        # 训练集保存
        df_train.to_csv(TRAIN_FILE_NAME, encoding='utf-8')
        print("-" * 50)
        print(f"💾 训练集数据已保存到：'{TRAIN_FILE_NAME}'")
        print(f"   训练集日期范围：{df_train.index.min()} - {df_train.index.max()}")
        print(f"   训练集大小: {len(df_train)} 个样本。")
        print(f"   训练集 Target 分布: {Counter(df_train['Target'])}")

        # 预测集保存
        df_predicting.to_csv(PREDICTING_FILE_NAME, encoding='utf-8')
        print("-" * 50)
        print(f"💾 预测集数据已保存到：'{PREDICTING_FILE_NAME}'")
        print(f"   预测集日期范围：{df_predicting.index.min()} - {df_predicting.index.max()}")
        print(f"   预测集大小: {len(df_predicting)} 个样本。")
        print("-" * 50)
        
        print("\n🎉 下一步：使用这两个文件进行 XGBoost/LightGBM 模型训练。")