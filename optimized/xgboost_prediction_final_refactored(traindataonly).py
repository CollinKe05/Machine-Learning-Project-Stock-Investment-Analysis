import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 忽略警告
warnings.filterwarnings("ignore")
np.random.seed(42)

# --- 1. 配置参数和文件路径 ---
TRAIN_FILE_NAME = "00700_train_data_final.csv"
PREDICTING_FILE_NAME = "00700_predicting_data_final.csv"
INITIAL_CAPITAL = 100000.0 

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 负号显示
# 🚀 Top 9 因子
FINAL_FEATURE_SET = [
    'Return_Lag_1', 'Return_Lag_5', 'Return_Lag_2', 
    'Daily_Return', 'Body_Ratio',      
    'MACD_HIST', 'MACD_DEA', 'MACD_DIF', 'RSI' 
]
TARGET_COLUMN = 'Target'

# ⚙️ 最终锁定 87.41% 收益的参数半仓逻辑
CONFIDENCE_THRESHOLD = 0.75   # 阈值在这里失效，但保留为 0.60
COOLING_PERIOD_DAYS = 0       # 约束条件
SELL_WEIGHT = 0.2             # 产生最佳收益的惩罚权重
SELL_WEIGHT_CANDIDATES = np.round(np.arange(0.0, 1.01, 0.1), 2)


# --- 2. 数据加载和预处理 (保持不变) ---
def load_and_prepare_data():
    global FINAL_FEATURE_SET 
    try:
        df_train = pd.read_csv(TRAIN_FILE_NAME, index_col='Date', parse_dates=True)
        df_predicting = pd.read_csv(PREDICTING_FILE_NAME, index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print(f"❌ 错误：未找到文件。")
        return None, None, None, None, None

    X_train = df_train[FINAL_FEATURE_SET]
    Y_train = df_train[TARGET_COLUMN]
    X_predicting = df_predicting[FINAL_FEATURE_SET]
    Y_train_mapped = Y_train.replace({-1: 0, 0: 1, 1: 2})
    print("-" * 50)
    print(f"✅ 数据加载成功！")
    print(f"训练集大小: {len(X_train)} 样本。")
    print(f"预测集大小: {len(X_predicting)} 样本。")
    print(f"使用的特征数量: {len(FINAL_FEATURE_SET)} 个。")
    print("-" * 50)
    return X_train, Y_train_mapped, X_predicting, Y_train, df_predicting 

def evaluate_sell_weight(
    sell_weight,
    X_train,
    Y_train_mapped,
    Y_train_original,
    df_train_raw,
    initial_capital=100000
):
    global SELL_WEIGHT
    SELL_WEIGHT = sell_weight

    # 1️⃣ 训练模型并预测训练集
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    weight_map = {0: SELL_WEIGHT, 1: 1.0, 2: 3.0}
    sample_weights = Y_train_mapped.map(weight_map)

    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        eval_metric='mlogloss',
        n_jobs=-1,
        seed=42
    )
    model.fit(X_scaled, Y_train_mapped, sample_weight=sample_weights)

    X_valid = df_train_raw[FINAL_FEATURE_SET]

    # === 用 pretrain 训练好的模型，预测 valid ===
    X_valid_scaled = scaler.transform(X_valid)
    proba = model.predict_proba(X_valid_scaled)

    pred_mapped = np.argmax(proba, axis=1)
    pred = (
        pd.Series(pred_mapped, index=df_train_raw.index)
        .replace({0: -1, 1: 0, 2: 1})
    )


    # 2️⃣ 用“训练集”做一次完整回测
    df_bt = df_train_raw.copy()
    df_bt['Predicted_Target'] = pred
    df_bt['Proba_1'] = proba[:, 2]

    df_bt, metrics = backtest_strategy(df_bt, initial_capital)

    # 3️⃣ 综合评分：收益 - 回撤惩罚
    score = metrics['Total_Strategy_Return'] - 0.5 * metrics['Max_Drawdown']

    return {
        'sell_weight': sell_weight,
        'return': metrics['Total_Strategy_Return'],
        'max_dd': metrics['Max_Drawdown'],
        'score': score
    }


# --- 3. XGBoost 模型训练与预测 (Sell 权重 0.1) ---

def train_and_predict_xgboost(X_train, Y_train_mapped, X_predicting, Y_train_original):
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_predicting_scaled = scaler.transform(X_predicting)
    
    # 🚀 Sell 权重 0.1
    weight_map = {0: SELL_WEIGHT, 1: 1.0, 2: 5.0} 
    sample_weights = Y_train_mapped.map(weight_map)
    
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob', num_class=3, n_estimators=1000, 
        learning_rate=0.03, max_depth=4, gamma=0.1, reg_lambda=0.5,            
        use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1, seed=42
    )

    print(f"📢 开始训练 XGBoost 模型 (Sell 权重 {SELL_WEIGHT})...")
    
    xgb_model.fit(X_train_scaled, Y_train_mapped, sample_weight=sample_weights)
    
    tscv = TimeSeriesSplit(n_splits=5)
    f1_scores = []

    for train_idx, val_idx in tscv.split(X_train_scaled):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = Y_train_mapped.iloc[train_idx], Y_train_mapped.iloc[val_idx]

        temp_model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            eval_metric='mlogloss',
            n_jobs=-1,
            seed=42
        )
        temp_model.fit(X_tr, y_tr)
        y_val_pred = temp_model.predict(X_val)
        f1_scores.append(f1_score(y_val, y_val_pred, average='macro'))

    print("-" * 50)
    print(f"📊 时间序列交叉验证 F1-Macro 均值: {np.mean(f1_scores):.4f}")
    print("-" * 50)

    # === 训练集最终评估（仅作参考） ===
    Y_train_pred_mapped = xgb_model.predict(X_train_scaled)
    Y_train_pred = pd.Series(Y_train_pred_mapped).replace({0: -1, 1: 0, 2: 1})

    print("📈 训练集分类报告：")
    print(classification_report(
        Y_train_original,
        Y_train_pred,
        target_names=['跌(-1)', '平(0)', '涨(1)']
    ))
    cm = confusion_matrix(Y_train_original, Y_train_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=['跌','平','涨'],
        yticklabels=['跌','平','涨'],
        cmap='Blues'
    )
    plt.title("训练集混淆矩阵")
    plt.xlabel("预测")
    plt.ylabel("真实")
    plt.tight_layout()
    plt.show()
    Y_predicting_proba = xgb_model.predict_proba(X_predicting_scaled)
    Y_predicting_pred_mapped = np.argmax(Y_predicting_proba, axis=1)
    Y_predicting_pred = pd.Series(Y_predicting_pred_mapped).replace({0: -1, 1: 0, 2: 1})
    Y_predicting_pred.index = X_predicting.index
    Y_predicting_pred.name = 'Predicted_Target'
    
    print("✅ 投资集预测完成 (已输出概率用于回撤控制)。")
    print(f"预测结果分布 (未过滤): {Counter(Y_predicting_pred)}")
    # ======================================
    # ✅ 新增：保存最终训练好的模型
    # ======================================
    MODEL_FILE_NAME = "final_xgb_model.json"
    try:
        xgb_model.save_model(MODEL_FILE_NAME)
        print(f"\n🎉 模型成功保存到文件: {MODEL_FILE_NAME}")
    except Exception as e:
        print(f"\n❌ 模型保存失败: {e}")
        
    # ======================================
    return Y_predicting_pred, pd.Series(Y_predicting_proba[:, 2], index=X_predicting.index, name='Proba_1')

# --- 4. 交易策略回测函数 (半仓买入逻辑) ---

def backtest_strategy(df, initial_capital):
    
    # 1. 信心阈值过滤买入信号
    df['Filtered_Signal'] = df.apply(
        lambda row: row['Predicted_Target'] 
                    if row['Predicted_Target'] == -1 or (row['Predicted_Target'] == 1 and row['Proba_1'] > CONFIDENCE_THRESHOLD) 
                    else 0,
        axis=1
    )
    
    df['Signal'] = df['Filtered_Signal'].shift(1) 
    df['Action'] = 0 
    
    capital = initial_capital
    position = 0.0
    portfolio_value = []
    
    last_action_index = -COOLING_PERIOD_DAYS - 1 
    
    for i, (index, row) in enumerate(df.iterrows()):
        
        current_value = capital + position * row['Close']
        portfolio_value.append(current_value) 

        signal = row['Signal']
        
        if pd.isna(signal):
            continue
            
        action = 0
        
        # 1. 检查冷却期
        if i - last_action_index <= COOLING_PERIOD_DAYS:
            action = 0 
        else:
            # 2. 执行交易逻辑 (!!! 关键：半仓买入)
            if signal == 1:  # 预测涨 且 信心足够：买入
                if capital > 0:
                    POSITION_RATIO = 0.5  # 半仓控制回撤
                    shares_to_buy = (capital * POSITION_RATIO) / row['Close']
                    position += shares_to_buy
                    capital -= capital * POSITION_RATIO

                    action = 1 
                    last_action_index = i 
            
            elif signal == -1: # 预测跌：卖出
                if position > 0:
                    capital += position * row['Close']
                    position = 0.0
                    action = -1 
                    last_action_index = i 
                
        df.loc[index, 'Action'] = action

    df['Portfolio_Value'] = portfolio_value 
    
    # --- 最终评估指标 (保持不变) ---
    final_value = df['Portfolio_Value'].iloc[-1]
    total_strategy_return = (final_value / initial_capital) - 1
    
    df['Peak'] = df['Portfolio_Value'].cummax()
    df['Drawdown'] = (df['Peak'] - df['Portfolio_Value']) / df['Peak']
    max_drawdown = df['Drawdown'].max()
    
    initial_price = df['Close'].iloc[0]
    final_price = df['Close'].iloc[-1]
    benchmark_return = (final_price / initial_price) - 1

    return df, {
        'Final_Value': final_value,
        'Total_Strategy_Return': total_strategy_return,
        'Max_Drawdown': max_drawdown,
        'Benchmark_Return': benchmark_return
    }

# --- 5. 结果可视化和输出 (保持不变) ---
def plot_results(df_results, metrics):
    df_results['Strategy_Equity'] = df_results['Portfolio_Value'] / df_results['Portfolio_Value'].iloc[0]
    df_results['Benchmark_Equity'] = df_results['Close'] / df_results['Close'].iloc[0]

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    
    plt.plot(df_results.index, df_results['Strategy_Equity'], label='ML 增强策略净值', color='blue', linewidth=2)
    plt.plot(df_results.index, df_results['Benchmark_Equity'], label='买入持有 (基准)', color='red', linestyle='--', linewidth=1)
    
    buy_signals = df_results[df_results['Action'] == 1].iloc[1:] 
    sell_signals = df_results[df_results['Action'] == -1].iloc[1:]

    ax.scatter(buy_signals.index, buy_signals['Strategy_Equity'], 
               marker='^', s=100, color='green', label='买入信号', alpha=1)
    ax.scatter(sell_signals.index, sell_signals['Strategy_Equity'], 
               marker='v', s=100, color='red', label='卖出信号', alpha=1)
    
    plt.title(f"投资组合净值曲线 (Sell惩罚 {SELL_WEIGHT}, 冷却期:{COOLING_PERIOD_DAYS}日, 半仓模式)")
    plt.xlabel("日期")
    plt.ylabel("净值")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- 6. 主程序运行 ---

if __name__ == "__main__":
    print("🔍 开始基于【收益 + 回撤】选择 Sell 权重...")
    results = []

    # ① 先加载数据
    X_train, Y_train_mapped, X_predicting, Y_train_original, df_predicting_raw = load_and_prepare_data()

    # ② 用训练集原始数据做回测
    df_train_raw = df_predicting_raw.loc[df_predicting_raw.index <= X_train.index.max()].copy()
    # ===== 新增：时间切分（不删原 df_train_raw）=====
    # ===== 按时间顺序比例切分（防止数据为空）=====
    split_point = int(len(df_train_raw) * 0.7)

    df_pretrain = pd.read_csv("00700_pretrain_data.csv", index_col='Date', parse_dates=True)
    df_valid    = pd.read_csv("00700_valid_data.csv", index_col='Date', parse_dates=True)


    # ===== 防御：避免预训练集为空 =====
    if len(df_pretrain) == 0:
        raise ValueError(
            "❌ df_pretrain 为空，请检查 TRAIN 数据起始日期，"
            "建议将切分点改为如 '2021-01-01'"
        )


    print("🔍 开始基于【收益 + 回撤】选择 Sell 权重...")
    results = []
    
    for w in SELL_WEIGHT_CANDIDATES:
        res = evaluate_sell_weight(
            w,
            X_train.loc[df_pretrain.index],
            Y_train_mapped.loc[df_pretrain.index],
            Y_train_original.loc[df_pretrain.index],
            df_valid              # ←【关键：回测用验证集】
        )

        results.append(res)
        print(f"Sell={w:.2f} | 收益={res['return']:.2%} | 回撤={res['max_dd']:.2%}")

    best = max(results, key=lambda x: x['score'])
    SELL_WEIGHT = best['sell_weight']

    print(f"\n✅ 最优 Sell 权重: {SELL_WEIGHT} (Score={best['score']:.4f})")

    X_train, Y_train_mapped, X_predicting, Y_train_original, df_predicting_raw = load_and_prepare_data()
    
    if X_train is not None:
        
        predicted_targets, predicted_proba_1 = train_and_predict_xgboost(
            X_train, Y_train_mapped, X_predicting, Y_train_original
        )
        
        df_predicting_raw['Predicted_Target'] = predicted_targets
        df_predicting_raw['Proba_1'] = predicted_proba_1 
        
        print("-" * 50)
        print(f"📢 开始进行交易策略回测 (冷却期: {COOLING_PERIOD_DAYS} 天, 信心阈值: {CONFIDENCE_THRESHOLD}, Sell惩罚: {SELL_WEIGHT}, 半仓交易)...")
        df_results, metrics = backtest_strategy(df_predicting_raw.copy(), INITIAL_CAPITAL)
        
        # 5. 输出评估结果
        print("-" * 50)
        print("📈 投资策略最终评估指标:")
        print(f"1. 初始资金: {INITIAL_CAPITAL:,.2f} CNY")
        print(f"2. 最终总资产: {metrics['Final_Value']:,.2f} CNY")
        print("-" * 50)
        print(f"3. 策略总收益率: {metrics['Total_Strategy_Return']:.2%}")
        print(f"4. **最终本金投资后收益率: {metrics['Total_Strategy_Return']:.2%}**") 
        print(f"5. 基准总收益率 (买入持有): {metrics['Benchmark_Return']:.2%}")
        print(f"6. **策略超额收益 (Alpha):** {(metrics['Total_Strategy_Return'] - metrics['Benchmark_Return']):.2%}")
        print("-" * 50)
        print(f"7. **最大回撤 (Max Drawdown):** {metrics['Max_Drawdown']:.2%}")
        print("-" * 50)
        # ===============================
        # 📊 测试集方向预测准确率（涨 vs 跌）
        # ===============================
        if 'Target' in df_predicting_raw.columns:
            df_eval = df_predicting_raw.copy()

            # 只保留真实为涨或跌的样本
            df_eval = df_eval[df_eval['Target'].isin([1, -1])]

            direction_acc = (
                df_eval['Target'] == df_eval['Predicted_Target']
            ).mean()

            print(f"📈 测试集涨跌方向预测准确率 (忽略平): {direction_acc:.2%}")
        else:
            print("⚠️ 测试集中无 Target，无法计算预测准确率")

        # 6. 可视化结果
        plot_results(df_results, metrics)
        
        # ===============================
        # 📊 交易一致性指标（只在有交易时）
        # ===============================
        df_trade = df_results[df_results['Action'] != 0].copy()

        if 'Target' in df_trade.columns and len(df_trade) > 0:
            trade_direction_acc = (
                np.sign(df_trade['Action']) ==
                np.sign(df_trade['Target'])
            ).mean()

            print(f"📈 交易方向一致率: {trade_direction_acc:.2%}")
        print(f"🎉 评估完成！")
