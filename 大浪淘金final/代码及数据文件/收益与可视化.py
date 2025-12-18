import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import load_model

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

FILE_PATH = '00700_cleaned.csv'
MODEL_PATH = 'best_model_combo9_score76.4.keras'

TRAIN_END = '2023-12-31'
TEST_START = '2024-01-01'
TEST_END='2025-04-24'
LOOKBACK = 15

def create_dataset(dataset_X, dataset_Y, look_back=1):
    X, Y = [], []
    for i in range(len(dataset_X) - look_back):
        X.append(dataset_X[i:(i + look_back)])
        Y.append(dataset_Y[i + look_back])
    return np.array(X), np.array(Y)

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
    df.reset_index(drop=True, inplace=True)

    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df = calculate_macd(df)
    df['Momentum'] = df['Close'] - df['Close'].shift(5)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def main():
    df = process_stock_data(FILE_PATH)

    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20', 'DIF', 'DEA', 'MACD_Hist']
    target_col = ['Close']

    train_df = df[df['Date'] <= TRAIN_END].copy()
    test_df_raw = df[(df['Date'] >= TEST_START) & (df['Date'] <= TEST_END)].copy()

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_Y = MinMaxScaler(feature_range=(0, 1))

    scaler_X.fit(train_df[feature_cols])
    scaler_Y.fit(train_df[target_col])

    full_test = pd.concat((train_df.iloc[-LOOKBACK:], test_df_raw))
    test_X_scaled_long = scaler_X.transform(full_test[feature_cols])

    test_X, _ = create_dataset(test_X_scaled_long, np.zeros(len(test_X_scaled_long)), LOOKBACK)

    model = load_model(MODEL_PATH)
    test_predict = model.predict(test_X, verbose=0)
    test_predict_real = scaler_Y.inverse_transform(test_predict)

    valid_len = min(len(test_df_raw), len(test_predict_real))
    result_df = test_df_raw.iloc[:valid_len].copy()
    result_df['Predicted'] = test_predict_real[:valid_len].flatten()

    y_true = result_df['Close'].values
    y_pred = result_df['Predicted'].values
    r2 = r2_score(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))

    cash = 100000
    position = 0
    assets = []

    buy_dates = []
    buy_prices = []
    sell_dates = []
    sell_prices = []

    result_df['Run_MA5'] = result_df['Close'].rolling(5).mean().fillna(method='bfill')

    for i in range(len(result_df) - 1):
        price = result_df.iloc[i]['Close']
        date = result_df.iloc[i]['Date']
        pred_next = result_df.iloc[i]['Predicted']
        ma5 = result_df.iloc[i]['Run_MA5']
        dif = result_df.iloc[i]['DIF']
        dea = result_df.iloc[i]['DEA']
        pred_ret = (pred_next - price) / price
        if position == 0:
            if (price > ma5) or (dif > dea) or (pred_ret > 0.01):
                shares = cash // price
                if shares > 0:
                    cash -= shares * price
                    position = shares
                    buy_dates.append(date)
                    buy_prices.append(price)
        elif position > 0:
            trend_bad = price < ma5
            macd_bad = dif < dea
            ai_panic = pred_ret < -0.015
            if (trend_bad and macd_bad) or ai_panic:
                cash += position * price
                position = 0
                sell_dates.append(date)
                sell_prices.append(price)
        assets.append(cash + position * price)

    assets.append(cash + position * result_df.iloc[-1]['Close'])
    result_df['Asset'] = assets

    final_asset = assets[-1]
    total_return = (final_asset - 100000) / 100000

    asset_series = pd.Series(assets)
    cummax = asset_series.cummax()
    drawdown = (asset_series - cummax) / cummax
    max_dd = drawdown.min()

    print(f"Metrics: ")
    print(f"  R^2: {r2:.4f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"\nTrading data: ")
    print(f"  Initial capital: 100,000.00")
    print(f"  Final assets: {final_asset:.2f}")
    print(f"  Total return: {total_return * 100:.2f}%")
    print(f"  Number of trades: {len(buy_dates) + len(sell_dates)} (buy + sell)")
    print(f"  Maximum drawdown: {max_dd * 100:.2f}%")


    plt.figure (figsize=(12, 6))
    plt.plot(result_df['Date'], result_df['Close'], label='Actual stock price', color='#1f77b4', alpha=0.6, linewidth=1.5)
    plt.plot(result_df['Date'], result_df['Predicted'], label='LSTM prediction', color='orange', linestyle='--', alpha=0.5)
    plt.title(f'Stock price trend', fontsize=14)
    plt.ylabel('Price')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

    plt.figure (figsize=(12, 6))
    plt.plot(result_df['Date'], result_df['Close'], label='Actual stock price', color='#1f77b4', alpha=0.6, linewidth=1.5)
    plt.plot(result_df['Date'], result_df['Predicted'], label='LSTM prediction', color='orange', linestyle='--', alpha=0.5)
    plt.scatter(buy_dates, buy_prices, marker='^', color='red', s=50, label='Buy signal', zorder=5)
    plt.scatter(sell_dates, sell_prices, marker='v', color='green', s=50, label='Sell signal', zorder=5)
    plt.title(f'Stock price trend and trading timing | Number of trades: {len(buy_dates)+len(sell_dates)}', fontsize=14)
    plt.ylabel('Price')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()


    plt.figure (figsize=(12, 6))
    plt.plot(result_df['Date'], result_df['Close'] / result_df['Close'].iloc[0], label='Benchmark (buy and hold)', color='gray',
             alpha=0.5, linestyle=':')
    plt.plot(result_df['Date'], result_df['Asset'] / 100000, label='Strategy net value', color='#d62728', linewidth=2.5)
    title_str = f'Final strategy | Return: {total_return * 100:.1f}%'
    plt.title(title_str, fontsize=14)
    plt.ylabel('Account net value (initial=1.0)')
    plt.xlabel('Date')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()


    plt.figure (figsize=(12, 6))
    plt.fill_between(result_df['Date'], drawdown*100, 0, color='red', alpha=0.3)
    plt.axhline(y=0, color='black', linewidth=0.5)
    plt.axhline(y=-10, color='orange', linestyle='--', alpha=0.5)
    plt.title(f'Drawdown analysis | Maximum drawdown: {max_dd * 100:.1f}%', fontsize=14)
    plt.grid()
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.show()


if __name__ == '__main__':
    main()