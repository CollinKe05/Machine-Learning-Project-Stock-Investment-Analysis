import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from collections import Counter
import warnings

# 忽略 pandas 的 SettingWithCopyWarning
warnings.filterwarnings("ignore")

# --- 1. 配置参数 ---
INPUT_FILE_NAME = "00700(1).txt"
CLEANED_CSV_NAME = "00700_preliminary_cleaned.csv"
TRAIN_END_DATE = '2023-12-31'
INVEST_START_DATE = '2024-01-01'
CLASSIFICATION_THRESHOLD = 0.005  # 涨跌阈值 delta = 0.5% 

# 最终选择的 Top 5 因子 (根据您上一步的 LightGBM 结果确定)
FINAL_FEATURE_SET = [
    'Return_Lag_1', 
    'Return_Lag_5', 
    'Return_Lag_2', 
    'Daily_Return', 
    'Body_Ratio'
]

# 中文到英文的列名映射字典
COLUMN_MAPPING = {
    '日期': 'Date', '开盘': 'Open', '最高': 'High', '最低': 'Low', 
    '收盘': 'Close', '成交量': 'Volume', '成交额': 'Amount'
}
encodings_to_try = ['gbk', 'gb18030', 'utf-8'] 

# --- 2. 文件读取和格式转换 ---

def load_and_preprocess_raw_data(file_name):
    """读取原始 TXT 文件，解决编码问题，并转换格式。"""
    df = None
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_name, sep='\t', header=0, skiprows=[0], encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            print(f"❌ 错误：未找到输入文件 '{file_name}'。")
            return None

    if df is None:
        return None

    original_cols = {col: col.strip() for col in df.columns}
    df.rename(columns=original_cols, inplace=True)
    df.rename(columns=COLUMN_MAPPING, inplace=True)
    
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount']
    df[price_cols] = df[price_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(how='all', inplace=True)
    
    return df

# --- 3. 完整的特征工程函数 (用于计算所有因子，以获得数据供选择) ---

def feature_engineering(df):
    """计算所有特征指标 (X) 和构建标签 (Y)。"""
    
    # 基础清洗与标签构建 (Y)
    df.fillna(method='ffill', inplace=True)
    df = df[df['Volume'] > 0].copy()
    
    df['Next_Day_Close'] = df['Close'].shift(-1)
    df['Next_Day_Return'] = (df['Next_Day_Close'] / df['Close']) - 1
    df['Target'] = 0  
    df.loc[df['Next_Day_Return'] > CLASSIFICATION_THRESHOLD, 'Target'] = 1
    df.loc[df['Next_Day_Return'] < -CLASSIFICATION_THRESHOLD, 'Target'] = -1
    
    df['Daily_Return'] = df['Close'].pct_change() 
    
    # --- 动量/趋势因子 ---
    SHORT_WINDOW, LONG_WINDOW, SIGNAL_WINDOW = 12, 26, 9
    df[f'SMA_{SHORT_WINDOW}'] = df['Close'].rolling(window=SHORT_WINDOW).mean()
    df[f'EMA_{SHORT_WINDOW}'] = df['Close'].ewm(span=SHORT_WINDOW, adjust=False).mean()
    df[f'EMA_{LONG_WINDOW}'] = df['Close'].ewm(span=LONG_WINDOW, adjust=False).mean()
    EMA_Short = df[f'EMA_{SHORT_WINDOW}']
    EMA_Long = df[f'EMA_{LONG_WINDOW}']
    df['MACD_DIF'] = EMA_Short - EMA_Long
    df['MACD_DEA'] = df['MACD_DIF'].ewm(span=SIGNAL_WINDOW, adjust=False).mean()
    df['MACD_HIST'] = df['MACD_DIF'] - df['MACD_DEA']

    # --- 超买超卖因子 (RSI) ---
    RSI_WINDOW = 14
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=RSI_WINDOW - 1, min_periods=RSI_WINDOW).mean()
    avg_loss = loss.ewm(com=RSI_WINDOW - 1, min_periods=RSI_WINDOW).mean()
    RS = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + RS))
    
    # --- 波动率因子 (BB & ATR) ---
    BB_WINDOW, BB_DEV, ATR_WINDOW = 20, 2, 14
    df['BB_Middle'] = df['Close'].rolling(window=BB_WINDOW).mean()
    df['StdDev'] = df['Close'].rolling(window=BB_WINDOW).std()
    df['BB_Upper'] = df['BB_Middle'] + (BB_DEV * df['StdDev'])
    df['BB_Lower'] = df['BB_Middle'] - (BB_DEV * df['StdDev'])
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle'] 

    df['High_PrevClose'] = abs(df['High'] - df['Close'].shift(1))
    df['Low_PrevClose'] = abs(df['Low'] - df['Close'].shift(1))
    df['True_Range'] = df['High'] - df['Low']
    df['TR'] = df[['True_Range', 'High_PrevClose', 'Low_PrevClose']].max(axis=1) 
    df['ATR'] = df['TR'].rolling(window=ATR_WINDOW).mean() 
    
    # --- 量价因子与滞后特征 ---
    LAG_N = 5 
    df['Volume_SMA_5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_5']

    for i in range(1, LAG_N + 1):
        df[f'Return_Lag_{i}'] = df['Daily_Return'].shift(i)

    # --- 补充高阶因子 ---
    STAT_WINDOW = 30
    high_low_diff = (df['High'] - df['Low']).replace(0, 1e-6) 
    mfm = ( (df['Close'] - df['Low']) - (df['High'] - df['Close']) ) / high_low_diff
    mfv = mfm * df['Volume']
    df['AD_Line'] = mfv.cumsum()

    obv_series = pd.Series(0, index=df.index)
    obv_series[df['Close'] > df['Close'].shift(1)] = df['Volume']
    obv_series[df['Close'] < df['Close'].shift(1)] = -df['Volume']
    df['OBV'] = obv_series.cumsum()
    
    df['Body_Length'] = abs(df['Close'] - df['Open'])
    df['Body_Ratio'] = (df['Body_Length'] / df['True_Range']).replace(np.inf, 1).fillna(0) 
    df['Upper_Wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Upper_Wick_Ratio'] = (df['Upper_Wick'] / df['True_Range']).replace(np.inf, 1).fillna(0)

    df['Close_vs_MA_Dev'] = (df['Close'] - df['BB_Middle']) / df['Close'] 
    df['Return_Skew'] = df['Daily_Return'].rolling(window=STAT_WINDOW).skew()
    df['Return_Kurt'] = df['Daily_Return'].rolling(window=STAT_WINDOW).kurt()

    df.dropna(inplace=True)

    return df

# --- 4. 因子评判函数 (保持原有结构，以确保流程完整性，但输出不用于最终选择) ---

def get_feature_importance(df):
    """提取所有因子的重要性，并返回包含所有特征的 X_train 和 Y_train。"""
    
    EXCLUDED_COLS = ['Next_Day_Close', 'Next_Day_Return', 'Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Amount']
    AUXILIARY_COLS = ['TR', 'High_PrevClose', 'Low_PrevClose', 'True_Range', 'Body_Length', 'Upper_Wick', 'Volume_SMA_5', 'StdDev', 'BB_Upper', 'BB_Lower']
    ALL_EXCLUDED_COLS = EXCLUDED_COLS + AUXILIARY_COLS
    
    FEATURE_COLUMNS = [col for col in df.columns if col not in ALL_EXCLUDED_COLS]
    
    X = df[FEATURE_COLUMNS]
    Y = df['Target']

    X_train_full = X.loc[X.index <= TRAIN_END_DATE]
    Y_train = Y.loc[Y.index <= TRAIN_END_DATE]
    
    if len(X_train_full) == 0:
        return None, None, FEATURE_COLUMNS

    # 训练模型并提取重要性
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    
    lgb_params = {
        'objective': 'multiclass', 'num_class': 3, 'boosting_type': 'gbdt',
        'n_estimators': 500, 'learning_rate': 0.05, 'verbose': -1, 'n_jobs': -1, 'seed': 42
    }
    
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_train_scaled, Y_train)
    
    importance_df = pd.DataFrame({
        'Feature': FEATURE_COLUMNS,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("-" * 50)
    print("📢 LightGBM 因子重要性评估 (Top 15 - 仅用于参考):")
    print(importance_df.head(15).to_markdown(index=False))
    print("-" * 50)
    
    # 返回完整的 X_train 和所有特征名
    return X_train_full, Y_train, FEATURE_COLUMNS


# --- 5. 主程序运行和最终特征选择 ---

if __name__ == "__main__":
    
    # 1. 读取和计算所有特征
    df_full = load_and_preprocess_raw_data(INPUT_FILE_NAME)
    
    if df_full is not None:
        df_with_features = feature_engineering(df_full.copy())
        
        # 2. 运行 LightGBM (可选：用于确认 Top 5 结果)
        X_train_full, Y_train, all_feature_names = get_feature_importance(df_with_features.copy())
        
        # 3. 最终特征选择和数据集划分
        X_final = df_with_features[FINAL_FEATURE_SET]
        Y_final = df_with_features['Target']
        
        X_train_final = X_final.loc[X_final.index <= TRAIN_END_DATE]
        Y_train_final = Y_final.loc[Y_final.index <= TRAIN_END_DATE]
        X_test_final = X_final.loc[X_final.index >= INVEST_START_DATE]

        print("-" * 50)
        print("✅ 最终特征选择完成！")
        print(f"👉 选取的 {len(FINAL_FEATURE_SET)} 个特征是：{FINAL_FEATURE_SET}")
        print(f"训练集 (X_train) 样本数: {X_train_final.shape[0]}, 特征数: {X_train_final.shape[1]}")
        print(f"投资集 (X_test) 样本数: {X_test_final.shape[0]}, 特征数: {X_test_final.shape[1]}")
        print(f"X_train_final 形状: {X_train_final.shape}")
        print(f"X_test_final 形状: {X_test_final.shape}")
        print("-" * 50)
        print("🎉 下一步：使用 X_train_final 和 Y_train_final 训练您的 LightGBM/XGBoost 模型。")