import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
import warnings
import matplotlib.pyplot as plt

# 忽略警告
warnings.filterwarnings("ignore")

# --- 1. 配置参数和文件路径 ---
TRAIN_FILE_NAME = "00700_train_data_final.csv"
PREDICTING_FILE_NAME = "00700_predicting_data_final.csv"
INITIAL_CAPITAL = 100000.0 
TRADE_LOG_FILE = "trade_log_rolling_strategy.csv" # 新增：交易日志输出文件

# 🚀 Top 9 因子
FINAL_FEATURE_SET = [
    'Return_Lag_1', 'Return_Lag_5', 'Return_Lag_2', 
    'Daily_Return', 'Body_Ratio',      
    'MACD_HIST', 'MACD_DEA', 'MACD_DIF', 'RSI' 
]
TARGET_COLUMN = 'Target'

# ⚙️ 激进策略调整：降低信心阈值 (从 0.60 -> 0.52)
CONFIDENCE_THRESHOLD = 0.52   # 信心阈值
COOLING_PERIOD_DAYS = 2       # 交易冷却期

# --- 2. 数据加载和预处理 (为滚动预测修改) ---

def load_full_data():
    """加载并合并训练集和预测集，以便进行滚动训练。"""
    
    try:
        # 加载整个训练集 (包含特征和Target)
        df_train = pd.read_csv(TRAIN_FILE_NAME, index_col='Date', parse_dates=True)
        # 加载整个预测集 (包含特征, Target 和 Close)
        df_predicting = pd.read_csv(PREDICTING_FILE_NAME, index_col='Date', parse_dates=True)
        
    except FileNotFoundError:
        print(f"❌ 错误：未找到文件。请确保 {TRAIN_FILE_NAME} 和 {PREDICTING_FILE_NAME} 文件存在。")
        return None, None
    
    # 检查特征完整性 (简化版，假设已修复)
    if not all(f in df_train.columns for f in FINAL_FEATURE_SET):
        print("⚠️ 警告：特征集不完整。请运行 data_splitting_final.py。")
        
    # 合并所有特征和标签 (用于滚动训练和预测)
    all_cols = FINAL_FEATURE_SET + [TARGET_COLUMN]
    df_full_features = pd.concat([df_train[all_cols], df_predicting[all_cols]])
    
    # 提取预测期原始数据 (用于回测逻辑，特别是Close价格)
    df_predicting_raw = df_predicting[['Close']].copy()

    # 确保 Target 转换为映射值
    df_full_features[TARGET_COLUMN + '_Mapped'] = df_full_features[TARGET_COLUMN].replace({-1: 0, 0: 1, 2: 2})
    
    print("-" * 50)
    print(f"✅ 数据加载成功！")
    print(f"总历史数据大小: {len(df_full_features)} 样本。")
    print(f"投资期样本数量: {len(df_predicting_raw)} 样本。")
    print("-" * 50)
    
    return df_full_features, df_predicting_raw

# --- 3. 核心：动态训练、预测和回测函数 (Rolling Walk-Forward) ---

def run_rolling_strategy(df_full_features, df_predicting_raw, initial_capital):
    
    invest_dates = df_predicting_raw.index
    
    # 初始化回测和日志变量
    capital = initial_capital
    position = 0.0
    portfolio_value = []
    last_action_index = -COOLING_PERIOD_DAYS - 1
    trade_log = []
    
    # 记录每天的信号和动作
    df_results = df_predicting_raw.copy()
    df_results['Action'] = 0 
    df_results['Predicted_Proba_1'] = np.nan # 记录每天的上涨概率
    df_results['Signal_For_Trade'] = 0 # 记录用于第二天交易的信号

    # 确定原始训练集截止索引 (第一个投资日期的前一个交易日)
    # 这确保我们从历史数据的第一个点开始训练
    first_invest_date_idx = df_full_features.index.get_loc(invest_dates[0])
    
    # 获取所有特征列和标签列
    feature_cols = FINAL_FEATURE_SET
    target_col = TARGET_COLUMN + '_Mapped'
    
    # 初始化 StandardScaler
    scaler = StandardScaler()

    print(f"📢 开始进行滚动训练和回测 (共 {len(invest_dates)} 交易日)...")
    
    # 滚动窗口迭代：从投资期第一天开始
    for i, current_date in enumerate(invest_dates):
        
        # 1. 定义动态训练集 (包含原始训练数据 + 所有已“解锁”的历史数据)
        # 训练集：从历史数据开始，到当前投资日期的前一个交易日 (iloc切片是独占末尾，因此切到 first_invest_date_idx + i 刚好包含 i-1 的数据)
        # 注意：i从0开始，所以第一个训练集切片大小为 first_invest_date_idx
        df_train_current = df_full_features.iloc[:first_invest_date_idx + i].copy()
        
        # 预测点：当前日期的特征 (预测明日的Target)
        X_predict_current = df_full_features.loc[[current_date], feature_cols]
        
        # 2. 训练模型 (每天重新训练)
        X_train_current = df_train_current[feature_cols]
        Y_train_current = df_train_current[target_col]
        
        # 每天重新拟合标准化 (反映数据分布的变化)
        X_train_scaled = scaler.fit_transform(X_train_current)
        X_predict_scaled = scaler.transform(X_predict_current)
        
        # 类别权重：2 (上涨) 权重 5.0
        weight_map = {0: 1.0, 1: 1.0, 2: 5.0} 
        sample_weights = Y_train_current.map(weight_map)
        
        # 为了加快滚动训练速度，减少 n_estimators 和 max_depth
        xgb_model = xgb.XGBClassifier(
            objective='multi:softprob', num_class=3, n_estimators=100, 
            learning_rate=0.1, max_depth=3, gamma=0.1, reg_lambda=0.5,            
            use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1, seed=42
        )
        xgb_model.fit(X_train_scaled, Y_train_current, sample_weight=sample_weights)
        
        # 3. 预测下一交易日信号
        Y_predicting_proba = xgb_model.predict_proba(X_predict_scaled)[0]
        proba_1 = Y_predicting_proba[2] # 上涨概率
        predicted_target_mapped = np.argmax(Y_predicting_proba)
        predicted_target = {0: -1, 1: 0, 2: 1}[predicted_target_mapped]

        # 4. 信心阈值过滤 (Signal T -> T+1)
        signal_for_next_day = predicted_target if predicted_target == -1 or (predicted_target == 1 and proba_1 > CONFIDENCE_THRESHOLD) else 0

        # 将上涨概率和信号记录到结果 DataFrame
        df_results.loc[current_date, 'Predicted_Proba_1'] = proba_1
        df_results.loc[current_date, 'Signal_For_Trade'] = signal_for_next_day
        
        # 5. 交易逻辑执行 (Trade T 使用 Signal T-1)
        # 提取用于今天交易的信号 (信号来自昨天)
        if i == 0:
            signal_to_act_on = 0 # 投资期第一天没有前一天的信号
        else:
            signal_to_act_on = df_results.loc[invest_dates[i-1], 'Signal_For_Trade']
        
        current_close = df_predicting_raw.loc[current_date, 'Close']

        action = 0
        
        # 检查冷却期 (冷却期跟踪的是索引 i)
        if i - last_action_index <= COOLING_PERIOD_DAYS:
            action = 0 
        else:
            # 交易执行
            if signal_to_act_on == 1:  # Buy
                if capital > 0:
                    shares_to_buy = capital / current_close
                    position += shares_to_buy
                    capital = 0.0
                    action = 1
                    last_action_index = i
                    trade_log.append({
                        'Date': current_date,
                        'Action': 'BUY',
                        'Price': current_close,
                        'Shares': shares_to_buy,
                        'Remaining_Capital': capital,
                        'Remaining_Shares': position
                    })
            
            elif signal_to_act_on == -1: # Sell
                if position > 0:
                    position_to_sell = position
                    capital += position_to_sell * current_close
                    position = 0.0
                    action = -1
                    last_action_index = i
                    trade_log.append({
                        'Date': current_date,
                        'Action': 'SELL',
                        'Price': current_close,
                        'Shares': -position_to_sell, # 卖出为负
                        'Remaining_Capital': capital,
                        'Remaining_Shares': position
                    })

        df_results.loc[current_date, 'Action'] = action
        
        # 6. 更新投资组合价值
        current_value = capital + position * current_close
        portfolio_value.append(current_value) 

    df_results['Portfolio_Value'] = portfolio_value 
    
    # --- 7. 评估指标和日志输出 ---
    final_value = df_results['Portfolio_Value'].iloc[-1]
    total_strategy_return = (final_value / initial_capital) - 1
    
    df_results['Peak'] = df_results['Portfolio_Value'].cummax()
    df_results['Drawdown'] = (df_results['Peak'] - df_results['Portfolio_Value']) / df_results['Peak']
    max_drawdown = df_results['Drawdown'].max()
    
    initial_price = df_results['Close'].iloc[0]
    final_price = df_results['Close'].iloc[-1]
    benchmark_return = (final_price / initial_price) - 1

    metrics = {
        'Final_Value': final_value,
        'Total_Strategy_Return': total_strategy_return,
        'Max_Drawdown': max_drawdown,
        'Benchmark_Return': benchmark_return
    }

    # 交易日志 CSV
    trade_log_df = pd.DataFrame(trade_log)
    if not trade_log_df.empty:
        trade_log_df.set_index('Date', inplace=True)
        trade_log_df.to_csv(TRADE_LOG_FILE)
        print(f"✅ 交易日志已保存到: {TRADE_LOG_FILE}")
    else:
        print("⚠️ 警告：交易日志为空，未发生任何交易。")

    return df_results, metrics

# --- 4. 结果可视化和输出 (不变) ---

def plot_results(df_results, metrics):
    """可视化策略净值和基准净值曲线，并标记买卖点。"""
    
    df_results['Strategy_Equity'] = df_results['Portfolio_Value'] / df_results['Portfolio_Value'].iloc[0]
    df_results['Benchmark_Equity'] = df_results['Close'] / df_results['Close'].iloc[0]

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    
    plt.plot(df_results.index, df_results['Strategy_Equity'], label='ML 增强策略净值', color='blue', linewidth=2)
    plt.plot(df_results.index, df_results['Benchmark_Equity'], label='买入持有 (基准)', color='red', linestyle='--', linewidth=1)
    
    buy_signals = df_results[df_results['Action'] == 1] 
    sell_signals = df_results[df_results['Action'] == -1]

    ax.scatter(buy_signals.index, buy_signals['Strategy_Equity'], 
               marker='^', s=100, color='green', label='买入信号', alpha=1)
    ax.scatter(sell_signals.index, sell_signals['Strategy_Equity'], 
               marker='v', s=100, color='red', label='卖出信号', alpha=1)
    
    plt.title(f"投资组合净值曲线 (滚动训练, 信心阈值:{CONFIDENCE_THRESHOLD}, 冷却期:{COOLING_PERIOD_DAYS}日)")
    plt.xlabel("日期")
    plt.ylabel("净值")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- 5. 主程序运行 ---

if __name__ == "__main__":
    
    df_full_features, df_predicting_raw = load_full_data()
    
    if df_full_features is not None:
        
        print("-" * 50)
        print(f"📢 开始进行滚动策略回测 (冷却期: {COOLING_PERIOD_DAYS} 天, 激进信心阈值: {CONFIDENCE_THRESHOLD})...")
        
        # 运行滚动回测
        # 注意：滚动回测会花费更多时间，因为它每天都需要重新训练模型。
        df_results, metrics = run_rolling_strategy(df_full_features.copy(), df_predicting_raw.copy(), INITIAL_CAPITAL)
        
        # 5. 输出评估结果
        print("-" * 50)
        print("📈 投资策略最终评估指标 (滚动训练模式, 激进策略):")
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
        
        # 6. 可视化结果
        plot_results(df_results, metrics)
        
        print(f"🎉 滚动策略运行完成！交易日志已保存到 {TRADE_LOG_FILE}。")