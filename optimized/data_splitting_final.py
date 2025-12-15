import pandas as pd
import numpy as np
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

# --- 1. 配置参数 (严格遵守项目要求) ---
INPUT_FILE_NAME = "00700(1).txt"
CLASSIFICATION_THRESHOLD = 0.005  # 涨跌阈值 delta = 0.5% 

# 项目定义的关键日期
TRAIN_END_DATE = '2023-12-31'
INVEST_START_DATE = '2024-01-01'
INVEST_END_DATE = '2025-04-24' 

# 🚀 优化方向一：更新为 Top 9 因子
FINAL_FEATURE_SET = [
    'Return_Lag_1', 'Return_Lag_5', 'Return_Lag_2', 
    'Daily_Return', 'Body_Ratio',
    # 新增的 MACD 和 RSI 因子
    'MACD_HIST', 'MACD_DEA', 'MACD_DIF', 'RSI' 
]
FINAL_COLUMNS = FINAL_FEATURE_SET + ['Target', 'Close']

# 输出文件名
TRAIN_FILE_NAME = "00700_train_data_final.csv"
PREDICTING_FILE_NAME = "00700_predicting_data_final.csv"

# 中文到英文的列名映射字典 (与之前相同)
COLUMN_MAPPING = {
    '日期': 'Date', '开盘': 'Open', '最高': 'High', '最低': 'Low', 
    '收盘': 'Close', '成交量': 'Volume', '成交额': 'Amount'
}
encodings_to_try = ['gbk', 'gb18030', 'utf-8'] 

# --- 2. 数据加载和特征工程函数 ---

def load_and_preprocess_raw_data(file_name):
    """加载原始数据并进行初步清洗和列名映射。"""
    df = None
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_name, sep='\t', header=0, skiprows=[0], encoding=encoding)
            break
        except Exception:
            continue
    if df is None:
        raise FileNotFoundError(f"❌ 错误：未找到或无法解析文件 '{file_name}'。")
        
    original_cols = {col: col.strip() for col in df.columns}
    df.rename(columns=original_cols, inplace=True)
    df.rename(columns=COLUMN_MAPPING, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount']
    df[price_cols] = df[price_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(how='all', inplace=True)
    return df.copy()

def feature_engineering_final(df):
    """计算所有 Top 9 因子所需的特征，以及 Target 标签。"""
    
    df.fillna(method='ffill', inplace=True)
    df = df[df['Volume'] > 0].copy()
    
    # Target (Y)
    df['Next_Day_Close'] = df['Close'].shift(-1)
    df['Next_Day_Return'] = (df['Next_Day_Close'] / df['Close']) - 1
    df['Target'] = 0  
    df.loc[df['Next_Day_Return'] > CLASSIFICATION_THRESHOLD, 'Target'] = 1
    df.loc[df['Next_Day_Return'] < -CLASSIFICATION_THRESHOLD, 'Target'] = -1
    
    # Daily_Return (R_t) 和滞后特征
    df['Daily_Return'] = df['Close'].pct_change() 
    LAG_N = 5
    for i in range(1, LAG_N + 1):
        df[f'Return_Lag_{i}'] = df['Daily_Return'].shift(i)

    # Body_Ratio (影线实体比)
    df['True_Range'] = df['High'] - df['Low']
    df['Body_Length'] = abs(df['Close'] - df['Open'])
    df['Body_Ratio'] = (df['Body_Length'] / df['True_Range']).replace(np.inf, 1).fillna(0) 

    # --- 🚀 新增：MACD 和 RSI 技术指标 (Top 9) ---
    # MACD (默认参数: 12, 26, 9)
    # EMA = Exponential Moving Average (指数移动平均)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_DIF'] = df['EMA_12'] - df['EMA_26']
    df['MACD_DEA'] = df['MACD_DIF'].ewm(span=9, adjust=False).mean()
    df['MACD_HIST'] = df['MACD_DIF'] - df['MACD_DEA']
    
    # RSI (默认参数: 14)
    delta = df['Close'].diff()
    # 分离涨幅和跌幅
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # 计算平均涨幅和平均跌幅 (使用 ewm 实现指数加权平均)
    avg_gain = gain.ewm(com=14 - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=14 - 1, adjust=False).mean()
    
    # 计算相对强度 RS 和 RSI
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df.dropna(inplace=True)
    return df

# --- 3. 主程序运行：特征提取、分割和保存 ---

if __name__ == "__main__":
    
    try:
        df_full = load_and_preprocess_raw_data(INPUT_FILE_NAME)
        df_with_features = feature_engineering_final(df_full)
        
        # 仅选择最终特征集和目标标签
        df_final = df_with_features[FINAL_COLUMNS].copy()
        
        # 严格按时间点分割数据集
        df_train = df_final.loc[df_final.index <= TRAIN_END_DATE].copy()
        df_predicting = df_final.loc[(df_final.index >= INVEST_START_DATE) & (df_final.index <= INVEST_END_DATE)].copy()

        # 确保预测集包含原始 Close 价格，以便回测
        df_predicting_output = df_with_features.loc[df_predicting.index, FINAL_COLUMNS + ['Close']].copy()

        # 保存为 CSV 文件
        df_train.to_csv(TRAIN_FILE_NAME, encoding='utf-8')
        df_predicting_output.to_csv(PREDICTING_FILE_NAME, encoding='utf-8')
        
        print("-" * 50)
        print("✅ 数据分割完成，现在包含 Top 9 因子！")
        print(f"💾 训练集 ('{TRAIN_FILE_NAME}') 大小: {len(df_train)} 样本。")
        print(f"💾 预测集 ('{PREDICTING_FILE_NAME}') 大小: {len(df_predicting_output)} 样本。")
        print("-" * 50)
        
    except Exception as e:
        print(f"发生错误: {e}")