import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

MODEL_PATH = 'best_model_combo9_score76.4.keras'
FILE_PATH = '00700_cleaned.csv'
LOOKBACK = 15
INITIAL_CASH = 100000
TEST_START = '2024-01-01'


# MACD
def calculate_macd(df):
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = exp1 - exp2
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = (df['DIF'] - df['DEA']) * 2
    return df


# MA5,MA20
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


def create_dataset(X, lookback):
    xs = []
    for i in range(len(X) - lookback):
        xs.append(X[i:i + lookback])
    return np.array(xs)


# Strategy 1
def strategy_buy_and_hold(df):
    cash = 100000
    assets = []

    first_price = df.iloc[0]['Close']
    shares = cash // first_price
    cash -= shares * first_price

    for i in range(len(df)):
        price = df.iloc[i]['Close']
        assets.append(cash + shares * price)
    return assets


# Strategy 2
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
    return assets


# Strategy 3
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
    return assets


# Strategy 4
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
    return assets


# Strategy 5
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
    return assets


# Strategy 6
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
    return assets


# Strategy 7
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
    return assets


# Strategy 8
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
    return assets


def calculate_max_drawdown(assets):
    if not assets:
        return 0.0

    peak = assets[0]
    max_drawdown = 0.0

    for asset in assets:
        if asset > peak:
            peak = asset
        drawdown = (peak - asset) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return max_drawdown


def main():
    model = load_model(MODEL_PATH)
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

    pred_scaled = model.predict(X_test, verbose=0)
    pred_price = scaler_Y.inverse_transform(pred_scaled)

    test_df = test_df.iloc[:len(pred_price)].copy()
    test_df['Predicted'] = pred_price.flatten()

    strategies = {
        "Buy and Hold": strategy_buy_and_hold,
        "Moving Average": strategy_no_ai,
        "Pure LSTM": strategy_ai_only,
        "Pure Direction Accuracy": strategy_ai_direction_only,
        "LSTM + MA5 + MACD ": strategy_original,
        "Weak LSTM": strategy_weak_ai,
        "Strict Strategy": strategy_strict_and,
        "LSTM + MA5": strategy_no_macd
    }

    results = {}
    for name, func in strategies.items():
        assets = func(test_df.copy())
        max_drawdown = calculate_max_drawdown(assets)
        results[name] = ((assets[-1] - 100000) / 100000, max_drawdown)

    sorted_results = sorted(results.items(), key=lambda x: (-x[1][0], x[1][1]))

    x = [result[1] * 100 for result in results.values()]
    y = [result[0] * 100 for result in results.values()]
    labels = list(results.keys())
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    plt.scatter(x, y, c=colors)
    for i, label in enumerate(labels):
        plt.annotate(label, (x[i], y[i]), fontsize=9, ha='center', va='top', xytext=(0, -8), textcoords='offset points')
    plt.xlabel('Maximum Drawdown (%)')
    plt.ylabel('Return Rate (%)')
    plt.title('Return Rate VS. Maximum Drawdown of Different Strategies')
    plt.grid(True)
    plt.show()

    for name, (ret, max_drawdown) in sorted_results:
        print(f"{name}: {ret * 100:.2f}%")

    if sorted_results:
        best_strategy, (best_ret, best_drawdown) = sorted_results[0]
        print(f"\nBest strategy: {best_strategy}")
        print(f"Best return rate: {best_ret * 100:.2f}%")
        print(f"Maximum drawdown of best strategy: {best_drawdown * 100:.2f}%")


if __name__ == '__main__':
    main()
