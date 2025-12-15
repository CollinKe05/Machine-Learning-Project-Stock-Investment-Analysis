import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from collections import Counter
import warnings
from sklearn.metrics import precision_recall_fscore_support

# 忽略 pandas 的 SettingWithCopyWarning
warnings.filterwarnings("ignore")

# --- 1. 配置参数 ---
# 假设原始文件名为 '00700(1).txt'
INPUT_FILE_NAME = "00700(1).txt"
CLEANED_CSV_NAME = "00700_preliminary_cleaned.csv"
TRAIN_END_DATE = '2023-12-31'
INVEST_START_DATE = '2024-01-01'
CLASSIFICATION_THRESHOLD = 0.005  # 涨跌阈值 delta = 0.5% (用于定义标签 Y)

# 中文到英文的列名映射字典
COLUMN_MAPPING = {
    '日期': 'Date', '开盘': 'Open', '最高': 'High', '最低': 'Low', 
    '收盘': 'Close', '成交量': 'Volume', '成交额': 'Amount'
}
encodings_to_try = ['gbk', 'gb18030', 'utf-8'] # 解决中文编码问题

# --- 2. 文件读取和格式转换 (结合之前的步骤) ---

def load_and_preprocess_raw_data(file_name):
    """读取原始 TXT 文件，解决编码问题，并转换格式。"""
    df = None
    print(f"尝试读取文件：{file_name}")
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(
                file_name, sep='\t', header=0, skiprows=[0], encoding=encoding
            )
            print(f"  - 成功使用 {encoding} 编码读取。")
            break
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            print(f"❌ 错误：未找到输入文件 '{file_name}'。")
            return None

    if df is None:
        print("❌ 转换失败：所有尝试的编码都无法正确解析文件。")
        return None

    # 清理列名和重命名
    original_cols = {col: col.strip() for col in df.columns}
    df.rename(columns=original_cols, inplace=True)
    df.rename(columns=COLUMN_MAPPING, inplace=True)
    
    # 基础类型转换和清洗
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount']
    df[price_cols] = df[price_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(how='all', inplace=True) # 删除全空行

    print(f"数据已转换为 {CLEANED_CSV_NAME}")
    # 保存为中间文件，方便后续调试
    df.to_csv(CLEANED_CSV_NAME, encoding='utf-8')
    return df

# --- 3. 完整的特征工程函数 ---

def feature_engineering(df):
    """计算所有特征指标 (X) 和构建标签 (Y)。"""
    
    # ==================================================
    # 1️⃣ 收益感知标签（替换原 Next_Day_Return 逻辑）
    # ==================================================
    FUTURE_WINDOW = 5          # 未来 5 日
    UP_THRESHOLD = 0.03        # +3%
    DOWN_THRESHOLD = -0.03    # -3%

    df['Future_Close'] = df['Close'].shift(-FUTURE_WINDOW)
    df['Future_Return'] = (df['Future_Close'] / df['Close']) - 1

    df['Target'] = 0
    df.loc[df['Future_Return'] > UP_THRESHOLD, 'Target'] = 1
    df.loc[df['Future_Return'] < DOWN_THRESHOLD, 'Target'] = -1

    print(
        f"\n收益感知标签 Target 构建完成 "
        f"(未来{FUTURE_WINDOW}日, ±{UP_THRESHOLD:.0%})，"
        f"标签分布：{Counter(df['Target'].dropna())}"
    )

    
    # 2. 计算日收益率和基础波动率指标 (用于后续因子计算)
    df['Daily_Return'] = df['Close'].pct_change() 
    df['True_Range'] = df['High'] - df['Low']
    
    # --- B. 动量/趋势因子 (SMA, EMA, MACD) ---
    SHORT_WINDOW = 12
    LONG_WINDOW = 26
    SIGNAL_WINDOW = 9

    df[f'SMA_{SHORT_WINDOW}'] = df['Close'].rolling(window=SHORT_WINDOW).mean()
    df[f'EMA_{SHORT_WINDOW}'] = df['Close'].ewm(span=SHORT_WINDOW, adjust=False).mean()
    df[f'EMA_{LONG_WINDOW}'] = df['Close'].ewm(span=LONG_WINDOW, adjust=False).mean()

    # MACD
    EMA_Short = df[f'EMA_{SHORT_WINDOW}']
    EMA_Long = df[f'EMA_{LONG_WINDOW}']
    df['MACD_DIF'] = EMA_Short - EMA_Long
    df['MACD_DEA'] = df['MACD_DIF'].ewm(span=SIGNAL_WINDOW, adjust=False).mean()
    df['MACD_HIST'] = df['MACD_DIF'] - df['MACD_DEA']

    # --- C. 超买超卖因子 (RSI) ---
    RSI_WINDOW = 14
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=RSI_WINDOW - 1, min_periods=RSI_WINDOW).mean()
    avg_loss = loss.ewm(com=RSI_WINDOW - 1, min_periods=RSI_WINDOW).mean()
    RS = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + RS))
    
    # --- D. 波动率因子 (Bollinger Bands & ATR) ---
    BB_WINDOW = 20
    BB_DEV = 2
    ATR_WINDOW = 14

    # 布林带宽度
    df['BB_Middle'] = df['Close'].rolling(window=BB_WINDOW).mean()
    df['StdDev'] = df['Close'].rolling(window=BB_WINDOW).std()
    df['BB_Upper'] = df['BB_Middle'] + (BB_DEV * df['StdDev'])
    df['BB_Lower'] = df['BB_Middle'] - (BB_DEV * df['StdDev'])
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle'] 

    # ATR
    df['High_PrevClose'] = abs(df['High'] - df['Close'].shift(1))
    df['Low_PrevClose'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['True_Range', 'High_PrevClose', 'Low_PrevClose']].max(axis=1) # 真实波幅 (TR)
    df['ATR'] = df['TR'].rolling(window=ATR_WINDOW).mean() 

    # --- E. 量价因子与滞后特征 ---
    LAG_N = 5 
    
    # 量比
    df['Volume_SMA_5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_5']

    # 滞后特征
    for i in range(1, LAG_N + 1):
        df[f'Return_Lag_{i}'] = df['Daily_Return'].shift(i)

    # --- F. 补充高阶因子 ---
    STAT_WINDOW = 30 # 用于统计计算的窗口期

    # 1. 累积/派发线 (A/D Line)
    mfm = ( (df['Close'] - df['Low']) - (df['High'] - df['Close']) ) / (df['High'] - df['Low']).replace(0, 1e-6)
    mfv = mfm * df['Volume']
    df['AD_Line'] = mfv.cumsum()

    # 2. 能量潮 (OBV)
    obv_series = pd.Series(0, index=df.index)
    obv_series[df['Close'] > df['Close'].shift(1)] = df['Volume']
    obv_series[df['Close'] < df['Close'].shift(1)] = -df['Volume']
    df['OBV'] = obv_series.cumsum()
    
    # 3. K线实体比 (Body Ratio) 和影线比
    df['Body_Length'] = abs(df['Close'] - df['Open'])
    df['Body_Ratio'] = (df['Body_Length'] / df['True_Range']).replace(np.inf, 1).fillna(0) 
    df['Upper_Wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Upper_Wick_Ratio'] = (df['Upper_Wick'] / df['True_Range']).replace(np.inf, 1).fillna(0)

    # 4. 收盘价与中轨 (MA) 的标准化偏差
    df['Close_vs_MA_Dev'] = (df['Close'] - df['BB_Middle']) / df['Close'] 

    # 5. 收益率的滚动统计量
    df['Return_Skew'] = df['Daily_Return'].rolling(window=STAT_WINDOW).skew()
    df['Return_Kurt'] = df['Daily_Return'].rolling(window=STAT_WINDOW).kurt()

    # --- G. 最终清理与划分 ---
    df.dropna(inplace=True)

    # 定义特征和标签列
    EXCLUDED_COLS = [
        'Future_Close',
        'Future_Return',
        'Next_Day_Close',
        'Next_Day_Return',
        'Target',
        'TR',
        'High_PrevClose',
        'Low_PrevClose',
        'Body_Length',
        'Upper_Wick',
        'Volume_SMA_5',
        'StdDev'
    ]

    EXCLUDED_COLS = [
        'Next_Day_Close', 'Next_Day_Return',
        'Future_Close', 'Future_Return',   # ← 新增
        'Target', 'TR', 'High_PrevClose', 'Low_PrevClose',
        'Body_Length', 'Upper_Wick', 'Volume_SMA_5', 'StdDev'
    ]

    X = df[FEATURE_COLUMNS]
    Y = df[LABEL_COLUMN]

    # 严格划分训练集和投资集
    X_train = X.loc[X.index <= TRAIN_END_DATE]
    Y_train = Y.loc[Y.index <= TRAIN_END_DATE]
    X_test = X.loc[X.index >= INVEST_START_DATE]

    print("-" * 50)
    print("✅ 特征工程完成！")
    print(f"最终特征数量 (X): {len(FEATURE_COLUMNS)} 个")
    print(f"训练集大小 (X_train/Y_train): {len(X_train)} 个样本 (日期范围: {X_train.index.min()} - {X_train.index.max()})")
    print(f"投资集大小 (X_test): {len(X_test)} 个样本")
    print("-" * 50)
    return X_train, Y_train, X_test, FEATURE_COLUMNS


# --- 4. 因子评判函数 (使用 LightGBM) ---

def evaluate_features_with_lgbm(X_train, Y_train, feature_names):
    """
    使用 LightGBM 在训练集上进行训练，并提取特征重要性。
    """
    print("📢 开始进行因子重要性评判 (基于 LightGBM)...")
    
    # 对特征进行标准化 (推荐操作，对树模型不是必须，但对后续模型融合有益)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 定义 LightGBM 参数
    # objective='multiclass': 因为是三分类问题 (1, 0, -1)
    lgb_params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': 3,
        'boosting_type': 'gbdt',
        'n_estimators': 500, # 迭代次数
        'learning_rate': 0.05,
        'feature_fraction': 0.8, # 随机选择特征比例，减少过拟合
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1, # 关闭输出
        'n_jobs': -1,
        'seed': 42
    }
    
    # 训练模型
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_train_scaled, Y_train)
    
    # 提取特征重要性
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("-" * 50)
    print("📊 因子重要性评估结果 (Top 15):")
    print(importance_df.head(15).to_markdown(index=False))
    print("-" * 50)
    
    # 评估模型性能（可选）
    cv_scores = cross_val_score(model, X_train_scaled, Y_train, cv=5, scoring='f1_macro', n_jobs=-1)
    print(f"模型在训练集上的 F1-Macro 5折交叉验证平均得分: {cv_scores.mean():.4f}")
    
    y_pred = model.predict(X_train_scaled)
    p, r, f, _ = precision_recall_fscore_support(Y_train, y_pred, average='macro')

    print(f"Precision (Macro): {p:.4f}")
    print(f"Recall (Macro): {r:.4f}")
    print(f"F1-Score (Macro): {f:.4f}")
    return importance_df


# --- 5. 主程序运行 ---

if __name__ == "__main__":
    # 1. 读取和预处理原始文件
    df_full = load_and_preprocess_raw_data(INPUT_FILE_NAME)
    
    if df_full is not None:
        # 2. 特征工程和数据集划分
        X_train, Y_train, X_test, feature_names = feature_engineering(df_full.copy())
        
        # 3. 因子评判
        if len(X_train) > 0:
            importance_df = evaluate_features_with_lgbm(X_train, Y_train, feature_names)
        else:
            print("⚠️ 训练集为空，无法进行因子评判。请检查数据日期和窗口期设置。")