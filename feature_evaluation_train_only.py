import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from collections import Counter
import warnings

warnings.filterwarnings("ignore")

# --- 1. 配置参数 ---
INPUT_FILE_NAME = "00700(1).txt"
TRAIN_END_DATE = '2023-12-31'
CLASSIFICATION_THRESHOLD = 0.005  # 涨跌阈值 delta = 0.5%

COLUMN_MAPPING = {
    '日期': 'Date', '开盘': 'Open', '最高': 'High', '最低': 'Low', 
    '收盘': 'Close', '成交量': 'Volume', '成交额': 'Amount'
}
encodings_to_try = ['gbk', 'gb18030', 'utf-8'] 

# --- 2. 数据加载和特征工程函数 (与之前一致，先计算所有特征) ---

def load_and_preprocess_raw_data(file_name):
    """读取原始 TXT 文件并转换为 DataFrame。"""
    # [此处省略与之前代码相同的 load_and_preprocess_raw_data 函数]
    # 请确保您已经将之前完整的 load_and_preprocess_raw_data 放入此文件中
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

def feature_engineering_all(df):
    """计算所有特征指标 (X) 和构建标签 (Y)。"""
    # [此处省略与之前代码相同的完整的 feature_engineering 函数 (包含所有33个特征)]
    # 此处省略的代码应该包含所有特征的计算逻辑。
    # 为了保证代码可运行，我将放入必要的代码块，请确保与您本地版本对齐。
    
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
    df['EMA_12'] = df['Close'].ewm(span=SHORT_WINDOW, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=LONG_WINDOW, adjust=False).mean()
    df['MACD_DIF'] = df['EMA_12'] - df['EMA_26']
    df['MACD_DEA'] = df['MACD_DIF'].ewm(span=SIGNAL_WINDOW, adjust=False).mean()
    df['MACD_HIST'] = df['MACD_DIF'] - df['MACD_DEA']

    # --- RSI ---
    RSI_WINDOW = 14
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    RS = gain.ewm(com=RSI_WINDOW - 1, min_periods=RSI_WINDOW).mean() / loss.ewm(com=RSI_WINDOW - 1, min_periods=RSI_WINDOW).mean()
    df['RSI'] = 100 - (100 / (1 + RS))
    
    # --- 波动率/量价/形态因子 (需要 Top 5 因子涉及的 Body_Ratio 依赖项) ---
    df['True_Range'] = df['High'] - df['Low']
    df['Body_Length'] = abs(df['Close'] - df['Open'])
    df['Body_Ratio'] = (df['Body_Length'] / df['True_Range']).replace(np.inf, 1).fillna(0) 
    
    LAG_N = 5
    for i in range(1, LAG_N + 1):
        df[f'Return_Lag_{i}'] = df['Daily_Return'].shift(i)
    
    # 添加其他必要的辅助特征，以匹配您之前33个特征的集合，这里只添加了最核心的
    df['Volume_SMA_5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_5']
    df['Return_Skew'] = df['Daily_Return'].rolling(window=30).skew()
    df['Return_Kurt'] = df['Daily_Return'].rolling(window=30).kurt()
    
    # 确保只保留所需的特征和标签
    df.dropna(inplace=True)
    return df

# --- 3. 因子评判函数 (核心：仅使用训练集数据) ---

def evaluate_features_with_lgbm(df_full_features):
    """
    使用 LightGBM 仅在训练集上进行训练，并提取特征重要性来评判因子。
    """
    # 1. 严格按时间点划分训练集
    df_train = df_full_features.loc[df_full_features.index <= TRAIN_END_DATE].copy()
    
    # 排除用于计算的中间列和原始价格列
    EXCLUDED_COLS = ['Next_Day_Close', 'Next_Day_Return', 'Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 
                     'True_Range', 'Body_Length', 'Volume_SMA_5'] # 排除辅助列

    FEATURE_COLUMNS = [col for col in df_train.columns if col not in EXCLUDED_COLS]
    
    X_train = df_train[FEATURE_COLUMNS]
    Y_train = df_train['Target']
    
    print("-" * 50)
    print("📢 开始进行因子重要性评判 (严格仅基于训练集 2018-2023)...")
    print(f"训练集大小: {len(X_train)} 样本。")
    print(f"参与评判的特征数量: {len(FEATURE_COLUMNS)} 个。")
    
    # 2. 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 3. 训练 LightGBM 模型
    lgb_params = {
        'objective': 'multiclass', 'num_class': 3, 'boosting_type': 'gbdt',
        'n_estimators': 1000, 'learning_rate': 0.05, 'verbose': -1, 'n_jobs': -1, 'seed': 42
    }
    
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_train_scaled, Y_train)
    
    # 4. 构建重要性 DataFrame
    importance_df = pd.DataFrame({
        'Feature': FEATURE_COLUMNS,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("-" * 50)
    print("✅ 因子重要性评估结果 (Top 15 - 严格无泄露):")
    print(importance_df.head(15).to_markdown(index=False))
    print("-" * 50)
    
    return importance_df


# --- 4. 主程序运行 ---

if __name__ == "__main__":
    
    try:
        df_full = load_and_preprocess_raw_data(INPUT_FILE_NAME)
        df_with_features = feature_engineering_all(df_full.copy())
        evaluate_features_with_lgbm(df_with_features)
        
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"发生错误: {e}")