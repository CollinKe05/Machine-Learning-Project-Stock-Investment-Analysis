import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import os
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False


# ================= 1. 设置随机种子 =================
def set_random_seeds(seed=40):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ================= 2. 参数设置 =================
FILE_PATH = '00700_cleaned.csv'
PRE_TRAIN_END = '2023-06-30'
VALID_START = '2023-07-01'
VALID_END = '2023-12-31'
TEST_START = '2024-01-01'
LOOKBACK = 15
EPOCHS = 60
BATCH_SIZE = 16
SEED = 40


# ================= 3. 数据处理 =================
def calculate_macd(df, fast=12, slow=26, signal=9):
    exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
    df['DIF'] = exp1 - exp2
    df['DEA'] = df['DIF'].ewm(span=signal, adjust=False).mean()
    df['MACD_Hist'] = (df['DIF'] - df['DEA']) * 2
    return df


def process_stock_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df = calculate_macd(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def create_dataset(X, Y, look_back):
    data_X, data_Y = [], []
    for i in range(len(X) - look_back):
        data_X.append(X[i:i + look_back])
        data_Y.append(Y[i + look_back])
    return np.array(data_X), np.array(data_Y)


# ================= 4. 回撤计算（⭐ 新增） =================
def calculate_max_drawdown(asset):
    asset = pd.Series(asset)
    cummax = asset.cummax()
    drawdown = (asset - cummax) / cummax
    return drawdown.min()


# ================= 5. 策略 =================
def run_strategy_with_signals(df, pred, params):
    INIT_CASH = 100000
    cash = INIT_CASH
    pos = 0
    assets = []
    trades = []

    df = df.copy()
    df['Predicted'] = pred
    df['Run_MA5'] = df['Close'].rolling(5).bfill()
    df['Run_MA20'] = df['Close'].rolling(20).bfill()

    for i in range(len(df) - 1):
        price = df.iloc[i]['Close']
        pnext = df.iloc[i]['Predicted']
        ma5 = df.iloc[i]['Run_MA5']
        ma20 = df.iloc[i]['Run_MA20']
        pret = (pnext - price) / price

        # === 买入 ===
        if pos == 0 and (
            pret > params['buy_ret'] and
            price > (ma20 if params['use_ma20'] else ma5)
        ):
            pos = cash // price
            cash -= pos * price
            trades.append((df.iloc[i]['Date'], 'BUY', price))

        # === 卖出 ===
        elif pos > 0 and (
            pret < params['sell_ret'] or
            price < (ma20 if params['use_ma20'] else ma5)
        ):
            cash += pos * price
            trades.append((df.iloc[i]['Date'], 'SELL', price))
            pos = 0

        assets.append(cash + pos * price)

    assets.append(cash + pos * df.iloc[-1]['Close'])

    ret = (assets[-1] - INIT_CASH) / INIT_CASH
    dd = max_drawdown(assets)

    return ret, dd, trades

STRATEGIES = {
    '长线投资': {
        'buy_ret': 0.005,
        'sell_ret': -0.03,
        'use_ma20': True
    },
    '短线投资': {
        'buy_ret': 0.015,
        'sell_ret': -0.01,
        'use_ma20': False
    },
    '稳健型投资': {
        'buy_ret': 0.008,
        'sell_ret': -0.02,
        'use_ma20': True
    },
    '激进型投资': {
        'buy_ret': 0.02,
        'sell_ret': -0.05,
        'use_ma20': False
    },
    '保守型投资': {
        'buy_ret': 0.004,
        'sell_ret': -0.015,
        'use_ma20': True
    }
}


# ================= 6. 训练 + 验证 =================
def train_with_validation():
    set_random_seeds(SEED)
    df = process_stock_data(FILE_PATH)

    pre_df = df[df['Date'] <= PRE_TRAIN_END]
    valid_df = df[(df['Date'] >= VALID_START) & (df['Date'] <= VALID_END)]
    test_df = df[df['Date'] >= TEST_START]

    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20', 'DIF', 'DEA', 'MACD_Hist']

    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    scaler_X.fit(pre_df[feature_cols])
    scaler_Y.fit(pre_df[['Close']])

    X_train, y_train = create_dataset(
        scaler_X.transform(pre_df[feature_cols]),
        scaler_Y.transform(pre_df[['Close']]),
        LOOKBACK
    )

    param_list = [
        (128, 64, 0.2, 0.001),
        (64, 32, 0.3, 0.001),
        (256, 128, 0.2, 0.0005),
        (128, 128, 0.3, 0.001),
        (64, 64, 0.4, 0.001),
        (128, 64, 0.3, 0.0005),
        (256, 256, 0.2, 0.001),
        (64, 32, 0.2, 0.0005),
        (128, 128, 0.4, 0.0005),
        (256, 128, 0.3, 0.001),
    ]

    records = []

    for idx, (u1, u2, dr, lr) in enumerate(param_list, 1):
        tf.keras.backend.clear_session()
        model = Sequential([
            LSTM(u1, return_sequences=True, input_shape=(LOOKBACK, len(feature_cols))),
            Dropout(dr),
            LSTM(u2),
            Dropout(dr),
            Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse')

        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

        full_valid = pd.concat((pre_df.iloc[-LOOKBACK:], valid_df))
        X_valid = scaler_X.transform(full_valid[feature_cols])
        X_valid, _ = create_dataset(X_valid, np.zeros(len(X_valid)), LOOKBACK)

        pred = scaler_Y.inverse_transform(model.predict(X_valid, verbose=0))
        valid_res = valid_df.iloc[:len(pred)].copy()
        valid_res['Predicted'] = pred.flatten()

        ret, dd = run_strategy_on_data(valid_res)
        r2 = r2_score(valid_res['Close'], valid_res['Predicted'])

        records.append({
            'model': model,
            'return': ret,
            'r2': r2,
            'drawdown': dd
        })

    # ================= ⭐ 核心修改：归一化 + 加权评分 =================
    result_df = pd.DataFrame(records)

    result_df['ret_n'] = (result_df['return'] - result_df['return'].min()) / (result_df['return'].max() - result_df['return'].min())
    result_df['r2_n'] = (result_df['r2'] - result_df['r2'].min()) / (result_df['r2'].max() - result_df['r2'].min())
    result_df['dd_n'] = (result_df['drawdown'] - result_df['drawdown'].min()) / (result_df['drawdown'].max() - result_df['drawdown'].min())
    result_df['dd_n'] = 1 - result_df['dd_n']  # 回撤越小越好

    # 权重（你之后可以随便改）
    W_RET, W_R2, W_DD = 0.3, 0.4, 0.3
    result_df['score'] = (
        W_RET * result_df['ret_n'] +
        W_R2 * result_df['r2_n'] +
        W_DD * result_df['dd_n']
    )

    best_row = result_df.sort_values('score', ascending=False).iloc[0]
    best_model = best_row['model']

    print("验证集最优模型：")
    print(best_row[['return', 'r2', 'drawdown', 'score']])

    # ================= 7. 测试集（完全不变） =================
    full_test = pd.concat((df[df['Date'] <= VALID_END].iloc[-LOOKBACK:], test_df))
    X_test = scaler_X.transform(full_test[feature_cols])
    X_test, _ = create_dataset(X_test, np.zeros(len(X_test)), LOOKBACK)

    test_pred = scaler_Y.inverse_transform(best_model.predict(X_test, verbose=0))
    test_res = test_df.iloc[:len(test_pred)].copy()
    test_res['Predicted'] = test_pred.flatten()
    print("\n===== 不同策略在测试集上的表现 =====")

    for name, params in STRATEGIES.items():
        ret, dd, trades = run_strategy_with_signals(
            test_df.iloc[:len(test_pred)],
            test_pred,
            params
        )

        print(f"\n【{name}】")
        print(f"收益率: {ret:.2%}")
        print(f"最大回撤: {dd:.2%}")
        print(f"交易次数: {len(trades)}")
        print("前5个交易信号:")
        for t in trades[:5]:
            print(t)


    test_return, test_dd = run_strategy_on_data(test_res)
    test_r2 = r2_score(test_res['Close'], test_res['Predicted'])

    print("\n===== 测试集结果 =====")
    print(f"收益率: {test_return:.2%}")
    print(f"最大回撤: {test_dd:.2%}")
    print(f"R2: {test_r2:.4f}")


if __name__ == '__main__':
    train_with_validation()
