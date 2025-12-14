import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score
)
import warnings
warnings.filterwarnings("ignore")

# --- 配置 ---
INPUT_FILE = "00700_clean_features.csv"
TRAIN_END = "2023-01-01"
VAL_END = "2024-01-01"

RANDOM_STATE = 42

FEATURES = [
    'Ret_Lag_1', 'Ret_Lag_2', 'Ret_Lag_5',
    'RSI_6', 'RSI_12',
    'MACD_Hist', 'Body_Ratio', 'Bias_20', 'ATR'
]

# 🔧 优化后的 XGBoost 参数（防过拟合）
XGB_PARAMS = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'eta': 0.05,               # 学习率降低
    'max_depth': 3,            # 减小树深度
    'min_child_weight': 5,     # 提高最小样本权重
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 1.0,         # L2 正则
    'reg_alpha': 0.1,          # L1 正则
    'random_state': RANDOM_STATE,
    'verbosity': 0
}
NUM_ROUND = 100  # 减少轮数，防止过拟合


def load_and_split_data():
    df = pd.read_csv(INPUT_FILE, parse_dates=['Date'], index_col='Date')
    df.sort_index(inplace=True)

    train_df = df[df.index < TRAIN_END].copy()
    val_df = df[(df.index >= TRAIN_END) & (df.index < VAL_END)].copy()
    test_df = df[df.index >= VAL_END].copy()

    print(f"📊 数据划分完成:")
    print(f"   训练集: {train_df.index.min().date()} ~ {train_df.index.max().date()} ({len(train_df)} 条)")
    print(f"   验证集: {val_df.index.min().date()} ~ {val_df.index.max().date()} ({len(val_df)} 条)")
    print(f"   回测集: {test_df.index.min().date()} ~ {test_df.index.max().date()} ({len(test_df)} 条)")

    return train_df, val_df, test_df


def scale_features(train_df, val_df, test_df):
    scaler = StandardScaler()
    train_df[FEATURES] = scaler.fit_transform(train_df[FEATURES])
    val_df[FEATURES] = scaler.transform(val_df[FEATURES])
    test_df[FEATURES] = scaler.transform(test_df[FEATURES])
    return train_df, val_df, test_df, scaler


def train_model_with_eval(X_train, y_train, X_val, y_val):
    # 映射标签：[-1, 0, 1] → [0, 1, 2]
    y_train_mapped = y_train.replace({-1: 0, 0: 1, 1: 2})
    y_val_mapped = y_val.replace({-1: 0, 0: 1, 1: 2})

    model = xgb.XGBClassifier(**XGB_PARAMS, n_estimators=NUM_ROUND)
    
    # ❌ XGBClassifier 不支持 eval_set 和 early_stopping_rounds
    # 只能用原生 API 来做训练曲线
    model.fit(X_train, y_train_mapped, verbose=False)
    return model


def plot_training_curve_with_manual_tracking(X_train, y_train, X_val, y_val):
    # 使用原生 API 来画训练曲线
    y_train_mapped = y_train.replace({-1: 0, 0: 1, 1: 2})
    y_val_mapped = y_val.replace({-1: 0, 0: 1, 1: 2})

    dtrain = xgb.DMatrix(X_train, label=y_train_mapped)
    dval = xgb.DMatrix(X_val, label=y_val_mapped)

    evals_result = {}
    bst = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=NUM_ROUND,
        evals=[(dtrain, 'train'), (dval, 'validation')],
        evals_result=evals_result,
        verbose_eval=False
    )

    # 绘制训练曲线
    plt.figure(figsize=(8, 5))
    plt.plot(evals_result['train']['mlogloss'], label='Train Loss', color='blue')
    plt.plot(evals_result['validation']['mlogloss'], label='Validation Loss', color='red')
    plt.xlabel('Boosting Round')
    plt.ylabel('Multi-class Log Loss')
    plt.title('Training vs Validation Loss Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('training_curve.png', dpi=150)
    plt.show()

    return bst, evals_result


def evaluate_on_set(model, df, name):
    X = df[FEATURES]
    y_true = df['Target']

    y_pred_mapped = model.predict(X)
    y_pred = pd.Series(y_pred_mapped).replace({0: -1, 1: 0, 2: 1}).values

    macro_f1 = f1_score(y_true, y_pred, average='macro', labels=[-1, 0, 1])
    acc = accuracy_score(y_true, y_pred)

    print(f"\n📈 {name} 集评估结果:")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   Macro-F1: {macro_f1:.4f}")
    print("\n详细分类报告:")
    print(classification_report(y_true, y_pred, target_names=['跌 (-1)', '震荡 (0)', '涨 (+1)'], labels=[-1, 0, 1]))

    return y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['跌', '震荡', '涨'],
        yticklabels=['跌', '震荡', '涨']
    )
    plt.title(f'{title} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{title.lower().replace(" ", "_")}.png', dpi=150)
    plt.show()


def backtest_strategy(test_df, y_pred):
    test_df = test_df.copy()
    test_df['Signal'] = y_pred  # [-1, 0, 1]

    # ⚠️ 关键：用今日信号 × 明日收益率（无未来信息）
    test_df['Tomorrow_Return'] = test_df['Close'].shift(-1) / test_df['Close'] - 1

    # 删除最后一行（无法计算明日收益）
    test_df = test_df.iloc[:-1].copy()

    # 策略收益 = 信号 * 明日收益率
    test_df['Strategy_Return'] = test_df['Signal'] * test_df['Tomorrow_Return']

    # 累计净值（处理 NaN）
    test_df['Strategy_Return'] = test_df['Strategy_Return'].fillna(0)
    test_df['Tomorrow_Return'] = test_df['Tomorrow_Return'].fillna(0)

    test_df['Cumulative_Strategy'] = (1 + test_df['Strategy_Return']).cumprod()
    test_df['Cumulative_BuyHold'] = (1 + test_df['Tomorrow_Return']).cumprod()

    # 绩效指标
    total_trades = (test_df['Signal'] != 0).sum()
    win_trades = ((test_df['Signal'] * test_df['Tomorrow_Return']) > 0).sum()
    win_rate = win_trades / total_trades if total_trades > 0 else 0
    total_profit = test_df['Strategy_Return'].sum()
    sharpe = test_df['Strategy_Return'].mean() / test_df['Strategy_Return'].std() * np.sqrt(252) \
             if test_df['Strategy_Return'].std() != 0 else 0

    print(f"\n💼 回测绩效摘要:")
    print(f"   总交易次数: {total_trades}")
    print(f"   胜率: {win_rate:.2%}")
    print(f"   总收益: {total_profit:.2%}")
    print(f"   年化夏普比率: {sharpe:.2f}")

    # 净值曲线
    plt.figure(figsize=(12, 6))
    plt.plot(test_df.index, test_df['Cumulative_Strategy'], label='策略净值', linewidth=2)
    plt.plot(test_df.index, test_df['Cumulative_BuyHold'], label='买入持有', linewidth=1, alpha=0.7)
    plt.title('📈 回测净值曲线 (2024年起)')
    plt.xlabel('日期')
    plt.ylabel('累计净值')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('equity_curve.png', dpi=150)
    plt.show()


def main():
    print("🚀 [Step 1] 加载并划分数据...")
    train_df, val_df, test_df = load_and_split_data()

    print(f"\n🔍 使用特征列: {FEATURES}")

    print("\n🔄 [Step 2] 特征标准化...")
    train_df, val_df, test_df, scaler = scale_features(train_df, val_df, test_df)

    print("\n🧠 [Step 3] 训练 XGBoost 模型...")
    X_train, y_train = train_df[FEATURES], train_df['Target']
    X_val, y_val = val_df[FEATURES], val_df['Target']

    model = train_model_with_eval(X_train, y_train, X_val, y_val)

    print("\n📉 [Step 4] 绘制训练曲线...")
    _, _ = plot_training_curve_with_manual_tracking(X_train, y_train, X_val, y_val)

    print("\n📊 [Step 5] 模型评估...")
    evaluate_on_set(model, train_df, "训练")
    evaluate_on_set(model, val_df, "验证")
    y_true_test, y_pred_test = evaluate_on_set(model, test_df, "回测")

    print("\n🧩 [Step 6] 混淆矩阵...")
    plot_confusion_matrix(y_true_test, y_pred_test, "回测集")

    print("\n💰 [Step 7] 策略回测...")
    backtest_strategy(test_df, y_pred_test)

    print("\n✅ 所有任务完成！图表已保存为 PNG 文件。")


if __name__ == "__main__":
    main()