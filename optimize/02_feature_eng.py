import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# --- 配置 ---
INPUT_FILE = "00700_clean.csv"
OUTPUT_FILE = "00700_clean_features.csv"  # 生成带特征的新文件

# =================================================================
# === 手动实现 TA-Lib 中的指标计算（纯 Pandas/NumPy）
# =================================================================

# 1. 相对强弱指数 (RSI)
def calculate_rsi(series, timeperiod):
    """手动计算 RSI"""
    diff = series.diff()
    gain = diff.mask(diff < 0, 0)
    loss = -diff.mask(diff > 0, 0)
    
    # 使用 EMA/SMMA 平滑
    def rma(x, n):
        a = 1/n
        return x.ewm(com=n - 1, adjust=False).mean()

    avg_gain = rma(gain, timeperiod)
    avg_loss = rma(loss, timeperiod)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 2. 移动平均收敛/发散指标 (MACD)
def calculate_macd(series, fastperiod=12, slowperiod=26, signalperiod=9):
    """手动计算 MACD"""
    ema_fast = series.ewm(span=fastperiod, adjust=False).mean()
    ema_slow = series.ewm(span=slowperiod, adjust=False).mean()
    
    diff = ema_fast - ema_slow # MACD 快线 (Diff)
    dea = diff.ewm(span=signalperiod, adjust=False).mean() # MACD 慢线 (Dea)
    macd = diff - dea # MACD 柱 (Hist)
    
    return diff, dea, macd

# 3. 平均真实波幅 (ATR)
def calculate_atr(high, low, close, timeperiod=14):
    """手动计算 ATR"""
    # 真实波幅 (TR) = Max[ (H - L), Abs(H - C_prev), Abs(L - C_prev) ]
    high_low = high - low
    high_close_prev = np.abs(high - close.shift(1))
    low_close_prev = np.abs(low - close.shift(1))
    
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # ATR 是 TR 的 EMA/SMMA 平滑
    atr = tr.ewm(com=timeperiod - 1, adjust=False).mean()
    return atr

# =================================================================

def feature_engineering():
    print(f"🚀 [Step 2] 开始特征工程...")
    df = pd.read_csv(INPUT_FILE, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    
    # 检查是否已有特征列（避免重复计算）
    existing_features = [col for col in df.columns if col.startswith(('Ret_Lag_', 'RSI_', 'MACD_', 'ATR', 'Body_', 'Bias_'))]
    if existing_features:
        print(f"⚠️ 检测到已有特征列: {existing_features}")
        print("   如需重新计算，请删除或重命名原始文件。")
        return
    
    # --- 1. 构造特征 (X) ---
    # ⚠️ 注意：所有特征必须基于当前行或之前的行，绝对不能用 shift(-1)
    
    # 基础收益率
    df['Returns'] = df['Close'].pct_change()
    
    # 滞后收益率 (Lag Features)
    for lag in [1, 2, 3, 5, 10]:
        df[f'Ret_Lag_{lag}'] = df['Returns'].shift(lag)
        
    # **手动计算** 动量指标 (RSI)
    df['RSI_6'] = calculate_rsi(df['Close'], timeperiod=6)
    df['RSI_12'] = calculate_rsi(df['Close'], timeperiod=12)
    
    # **手动计算** 趋势指标 (MACD)
    diff, dea, macd = calculate_macd(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD_Diff'] = diff
    df['MACD_Dea'] = dea
    df['MACD_Hist'] = macd
    
    # **手动计算** 波动率 (ATR)
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], timeperiod=14)
    
    # 情绪指标：K线实体比 (Body Ratio)
    # (收 - 开) / (高 - 低)
    df['Body_Ratio'] = (df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-9)
    
    # 均线偏离度
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Bias_20'] = (df['Close'] - df['MA20']) / df['MA20']
    
    # --- 2. 构造标签 (Y) ---
    # ⚠️ 标签必须是“未来”的。我们预测的是 T+1 的收益。
    # 逻辑：如果 明天的收盘价 > 今天的收盘价 * (1+阈值)，则涨
    
    THRESHOLD = 0.005 # 0.5% 的涨跌阈值
    df['Next_Ret'] = df['Close'].shift(-1) / df['Close'] - 1
    
    conditions = [
        (df['Next_Ret'] > THRESHOLD),
        (df['Next_Ret'] < -THRESHOLD)
    ]
    choices = [1, -1] # 1: 涨, -1: 跌, 0: 震荡
    df['Target'] = np.select(conditions, choices, default=0)
    
    # --- 3. 清洗空值 ---
    # 由于计算 RSI/MACD/ATR 时，前几十行会有 NaN，这里一起清除。
    df.dropna(inplace=True) 
    
    # --- 4. 保存 ---
    df.to_csv(OUTPUT_FILE)
    
    print(f"✅ 特征工程完成，已保存至: {OUTPUT_FILE}")
    print(f"📊 新增特征列: {[col for col in df.columns if col.startswith(('Ret_Lag_', 'RSI_', 'MACD_', 'ATR', 'Body_', 'Bias_'))]}")
    print(f"📊 数据维度: {df.shape}")
    
    # 显示前几行以验证
    print("\n📋 前5行数据预览:")
    print(df.head())

if __name__ == "__main__":
    feature_engineering()