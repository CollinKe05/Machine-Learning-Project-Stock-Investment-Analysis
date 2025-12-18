import pandas as pd
import numpy as np
import math
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

#基本参数
MODEL_PATH = 'best_model_top10_combo9_score76.4.keras'
FILE_PATH = '00700_cleaned.csv'
LOOKBACK = 15
INITIAL_CASH = 100000
TEST_START = '2024-01-01'

#MACD
def calculate_macd(df):
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = exp1 - exp2
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = (df['DIF'] - df['DEA']) * 2
    return df

#MA5,MA20
def process_data(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)

    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df = calculate_macd(df)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

#滚动窗口
def create_dataset(X, lookback):
    xs = []
    for i in range(len(X) - lookback):
        xs.append(X[i:i + lookback])
    return np.array(xs)

#策略1：持股
def strategy_buy_and_hold(df):
    cash = 100000
    assets = []

    first_price = df.iloc[0]['Close']
    shares = cash // first_price
    cash -= shares * first_price

    for i in range(len(df)):
        price = df.iloc[i]['Close']
        assets.append(cash + shares * price)

    final_return = (assets[-1] - 100000) / 100000
    return final_return

#策略2：均线
def strategy_no_ai(df):
    cash = 100000
    position = 0
    assets = []

    df['Run_MA5'] = df['Close'].rolling(5).mean().bfill()

    for i in range(len(df) - 1):
        price = df.iloc[i]['Close']
        ma5 = df.iloc[i]['Run_MA5']
        dif = df.iloc[i]['DIF']
        dea = df.iloc[i]['DEA']

        if position == 0:
            if (price > ma5) or (dif > dea):
                shares = cash // price
                cash -= shares * price
                position = shares
        else:
            if (price < ma5 and dif < dea):
                cash += position * price
                position = 0

        assets.append(cash + position * price)

    assets.append(cash + position * df.iloc[-1]['Close'])
    return (assets[-1] - 100000) / 100000

#策略3：纯lstm预测
def strategy_ai_only(df):
    cash = 100000
    position = 0
    assets = []

    for i in range(len(df) - 1):
        price = df.iloc[i]['Close']
        pred = df.iloc[i]['Predicted']
        pred_ret = (pred - price) / price

        if position == 0 and pred_ret > 0.01:
            shares = cash // price
            cash -= shares * price
            position = shares

        elif position > 0 and pred_ret < -0.015:
            cash += position * price
            position = 0

        assets.append(cash + position * price)

    assets.append(cash + position * df.iloc[-1]['Close'])
    return (assets[-1] - 100000) / 100000

#策略4：纯方向准确率
def strategy_ai_direction_only(df):
    cash = 100000
    position = 0
    assets = []

    for i in range(len(df) - 1):
        price = df.iloc[i]['Close']
        pred = df.iloc[i]['Predicted']

        if position == 0 and pred > price:
            shares = cash // price
            if shares > 0:
                cash -= shares * price
                position = shares

        elif position > 0 and pred < price:
            cash += position * price
            position = 0

        assets.append(cash + position * price)

    assets.append(cash + position * df.iloc[-1]['Close'])
    return (assets[-1] - 100000) / 100000

# 策略5：lstm预测+趋势+MACD
def strategy_original(df):
    cash = 100000
    position = 0
    assets = []

    df['Run_MA5'] = df['Close'].rolling(5).mean().bfill()

    for i in range(len(df) - 1):
        price = df.iloc[i]['Close']
        pred_next = df.iloc[i]['Predicted']
        ma5 = df.iloc[i]['Run_MA5']
        dif = df.iloc[i]['DIF']
        dea = df.iloc[i]['DEA']
        pred_ret = (pred_next - price) / price

        if position == 0:
            cond1 = price > ma5
            cond2 = dif > dea
            cond3 = pred_ret > 0.01
            if cond1 or cond2 or cond3:
                shares = cash // price
                if shares > 0:
                    cash -= shares * price
                    position = shares

        else:
            trend_bad = price < ma5
            macd_bad = dif < dea
            ai_panic = pred_ret < -0.015
            if (trend_bad and macd_bad) or ai_panic:
                cash += position * price
                position = 0

        assets.append(cash + position * price)

    assets.append(cash + position * df.iloc[-1]['Close'])
    return (assets[-1] - 100000) / 100000

#策略六：弱lstm
def strategy_weak_ai(df):
    cash = 100000
    position = 0
    assets = []

    df['Run_MA5'] = df['Close'].rolling(5).mean().bfill()

    for i in range(len(df) - 1):
        price = df.iloc[i]['Close']
        pred = df.iloc[i]['Predicted']
        ma5 = df.iloc[i]['Run_MA5']
        dif = df.iloc[i]['DIF']
        dea = df.iloc[i]['DEA']
        pred_ret = (pred - price) / price

        if position == 0:
            if (price > ma5) or (dif > dea) or (pred_ret > 0.03):
                shares = cash // price
                cash -= shares * price
                position = shares
        else:
            if ((price < ma5 and dif < dea) or (pred_ret < -0.03)):
                cash += position * price
                position = 0

        assets.append(cash + position * price)

    assets.append(cash + position * df.iloc[-1]['Close'])
    return (assets[-1] - 100000) / 100000

#策略7：严苛
def strategy_strict_and(df):
    cash = 100000
    position = 0
    assets = []

    df['Run_MA5'] = df['Close'].rolling(5).mean().bfill()

    for i in range(len(df) - 1):
        price = df.iloc[i]['Close']
        pred = df.iloc[i]['Predicted']
        ma5 = df.iloc[i]['Run_MA5']
        dif = df.iloc[i]['DIF']
        dea = df.iloc[i]['DEA']
        pred_ret = (pred - price) / price

        if position == 0:
            if (price > ma5) and (dif > dea) and (pred_ret > 0.01):
                shares = cash // price
                cash -= shares * price
                position = shares
        else:
            if (price < ma5 and dif < dea) and (pred_ret < -0.015):
                cash += position * price
                position = 0

        assets.append(cash + position * price)

    assets.append(cash + position * df.iloc[-1]['Close'])
    return (assets[-1] - 100000) / 100000

#策略八：去MACD
def strategy_no_macd(df):
    cash = 100000
    position = 0
    assets = []

    df['Run_MA5'] = df['Close'].rolling(5).mean().bfill()

    for i in range(len(df) - 1):
        price = df.iloc[i]['Close']
        pred = df.iloc[i]['Predicted']
        ma5 = df.iloc[i]['Run_MA5']
        pred_ret = (pred - price) / price

        if position == 0:
            if (price > ma5) or (pred_ret > 0.01):
                shares = cash // price
                cash -= shares * price
                position = shares
        else:
            if (price < ma5) or (pred_ret < -0.015):
                cash += position * price
                position = 0

        assets.append(cash + position * price)

    assets.append(cash + position * df.iloc[-1]['Close'])
    return (assets[-1] - 100000) / 100000

#main
def main():
    print(">>> 加载模型")
    model = load_model(MODEL_PATH)

    print(">>> 读取并处理数据")
    df = process_data(FILE_PATH)

    train_df = df[df['Date'] < TEST_START]
    test_df = df[df['Date'] >= TEST_START].copy()

    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'MA5', 'MA20', 'DIF', 'DEA', 'MACD_Hist'
    ]

    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    scaler_X.fit(train_df[feature_cols])
    scaler_Y.fit(train_df[['Close']])

    full_test = pd.concat([train_df.iloc[-LOOKBACK:], test_df])
    X_test = scaler_X.transform(full_test[feature_cols])
    X_test = create_dataset(X_test, LOOKBACK)

    print(">>> 模型预测测试集")
    pred_scaled = model.predict(X_test, verbose=0)
    pred_price = scaler_Y.inverse_transform(pred_scaled)

    test_df = test_df.iloc[:len(pred_price)].copy()
    test_df['Predicted'] = pred_price.flatten()

    strategies = {
        "持股": strategy_buy_and_hold,
        "均线策略": strategy_no_ai,
        "纯lstm预测": strategy_ai_only,
        "纯方向准确率": strategy_ai_direction_only,
        "lstm + MA5 + MACD ": strategy_original,
        "弱lstm": strategy_weak_ai,
        "严格策略": strategy_strict_and,
        "去MACD": strategy_no_macd
    }

    print("\n不同策略在测试集上的收益率（从高到低排序）：")
    results = {}
    for name, func in strategies.items():
        ret = func(test_df.copy())
        results[name] = ret

    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    for name, ret in sorted_results:
        print(f"{name}: {ret * 100:.2f}%")


    if sorted_results:
        best_strategy, best_return = sorted_results[0]
        print(f"\n最佳策略: {best_strategy}")
        print(f"最佳收益率: {best_return * 100:.2f}%")
    else:
        print("\n未找到有效的策略结果")


if __name__ == '__main__':
    main()