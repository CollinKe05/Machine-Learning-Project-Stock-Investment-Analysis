import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

# =============================
# 1️⃣ 参数区
# =============================
TRAIN_FILE = "00700_train_data_final.csv"
PREDICT_FILE = "00700_predicting_data_final.csv"

INITIAL_CAPITAL = 100000
CONFIDENCE_THRESHOLD = 0.75
POSITION_RATIO = 0.5

SELL_WEIGHT_GRID = np.arange(0.0, 1.01, 0.1)

FEATURES = [
    'Return_Lag_1','Return_Lag_2','Return_Lag_5',
    'Daily_Return','Body_Ratio',
    'MACD_HIST','MACD_DEA','MACD_DIF','RSI'
]

# =============================
# 2️⃣ 数据加载 & 切分
# =============================
def load_data():
    df = pd.read_csv(TRAIN_FILE, index_col='Date', parse_dates=True)

    df_pretrain = df.loc[:'2022-12-31']
    df_valid    = df.loc['2023-01-01':'2023-12-31']

    return df_pretrain, df_valid


# =============================
# 3️⃣ 回测函数（简化但稳定）
# =============================
def backtest(df, capital=100000):
    pos = 0
    cash = capital
    pv = []

    for i in range(1, len(df)):
        price = df['Close'].iloc[i]
        signal = df['Signal'].iloc[i-1]

        if signal == 1 and cash > 0:
            buy_cash = cash * POSITION_RATIO
            pos += buy_cash / price
            cash -= buy_cash

        elif signal == -1 and pos > 0:
            cash += pos * price
            pos = 0

        pv.append(cash + pos * price)

    if len(pv) == 0:
        return np.nan, np.nan

    pv = pd.Series(pv)
    ret = pv.iloc[-1] / capital - 1
    dd = (pv.cummax() - pv).max() / pv.cummax().max()

    return ret, dd


# =============================
# 4️⃣ Sell weight 选择（仅用 2023）
# =============================
def select_sell_weight(df_train, df_valid):
    X_tr = df_train[FEATURES]
    y_tr = df_train['Target'].replace({-1:0,0:1,1:2})

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)

    best = None
    records = []

    for w in SELL_WEIGHT_GRID:
        weights = y_tr.map({0:w,1:1,2:3})

        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            eval_metric='mlogloss',
            n_jobs=-1,
            seed=42
        )
        model.fit(X_tr, y_tr, sample_weight=weights)

        # === 在验证集上回测 ===
        X_val = scaler.transform(df_valid[FEATURES])
        proba = model.predict_proba(X_val)
        pred = np.argmax(proba, axis=1)
        pred = pd.Series(pred).replace({0:-1,1:0,2:1}).values

        df_bt = df_valid.copy()
        df_bt['Signal'] = np.where(
            (pred == 1) & (proba[:,2] > CONFIDENCE_THRESHOLD), 1,
            np.where(pred == -1, -1, 0)
        )

        ret, dd = backtest(df_bt, INITIAL_CAPITAL)
        if np.isnan(ret):
            continue

        score = ret - 0.5 * dd
        records.append((w, ret, dd, score))

        if best is None or score > best[-1]:
            best = (w, ret, dd, score)

    print("\n📊 Sell Weight Grid Result:")
    for r in records:
        print(f"Sell={r[0]:.1f} | 收益={r[1]:.2%} | 回撤={r[2]:.2%} | Score={r[3]:.4f}")

    print(f"\n✅ 最优 Sell Weight = {best[0]:.2f}")
    return best[0]


# =============================
# 5️⃣ 主流程（无数据泄露）
# =============================
if __name__ == "__main__":
    df_pretrain, df_valid = load_data()

    best_sell = select_sell_weight(df_pretrain, df_valid)

    print("\n🚀 接下来：")
    print("1️⃣ 用 ≤2023 全部数据训练模型")
    print("2️⃣ 2024+ 仅做一次实战回测（不再调参）")
