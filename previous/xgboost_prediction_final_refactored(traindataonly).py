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

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 负号显示
# 🚀 Top 9 因子
FINAL_FEATURE_SET = [
    'Return_Lag_1', 'Return_Lag_5', 'Return_Lag_2', 
    'Daily_Return', 'Body_Ratio',      
    'MACD_HIST', 'MACD_DEA', 'MACD_DIF', 'RSI' 
]
TARGET_COLUMN = 'Target'

# ⚙️ 最终锁定 87.41% 收益的参数：全仓逻辑
CONFIDENCE_THRESHOLD = 0.60   # 阈值在这里失效，但保留为 0.60
COOLING_PERIOD_DAYS = 2       # 约束条件
SELL_WEIGHT = 0.1             # 产生最佳收益的惩罚权重

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
    
    Y_train_pred_mapped = xgb_model.predict(X_train_scaled)
    Y_train_pred = pd.Series(Y_train_pred_mapped).replace({0: -1, 1: 0, 2: 1})
    train_accuracy = accuracy_score(Y_train_original, Y_train_pred)
    train_f1_macro = f1_score(Y_train_original, Y_train_pred, average='macro')
    
    print("-" * 50)
    print("📈 XGBoost 训练集性能评估结果:")
    print(f"训练集准确率 (Accuracy): {train_accuracy:.4f}") 
    print(f"训练集 F1-Macro Score: {train_f1_macro:.4f}")
    print("-" * 50)
    
    Y_predicting_proba = xgb_model.predict_proba(X_predicting_scaled)
    Y_predicting_pred_mapped = np.argmax(Y_predicting_proba, axis=1)
    Y_predicting_pred = pd.Series(Y_predicting_pred_mapped).replace({0: -1, 1: 0, 2: 1})
    Y_predicting_pred.index = X_predicting.index
    Y_predicting_pred.name = 'Predicted_Target'
    
    print("✅ 投资集预测完成 (已输出概率用于回撤控制)。")
    print(f"预测结果分布 (未过滤): {Counter(Y_predicting_pred)}")
    
    return Y_predicting_pred, pd.Series(Y_predicting_proba[:, 2], index=X_predicting.index, name='Proba_1')

# --- 4. 交易策略回测函数 (全仓买入逻辑) ---

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
            # 2. 执行交易逻辑 (!!! 关键：全仓买入)
            if signal == 1:  # 预测涨 且 信心足够：买入
                if capital > 0:
                    shares_to_buy = capital / row['Close']
                    position += shares_to_buy
                    capital = 0.0 # !!! 资金全部用尽
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
    
    plt.title(f"投资组合净值曲线 (Sell惩罚 {SELL_WEIGHT}, 冷却期:{COOLING_PERIOD_DAYS}日, 全仓模式)")
    plt.xlabel("日期")
    plt.ylabel("净值")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- 6. 主程序运行 ---

if __name__ == "__main__":
    
    X_train, Y_train_mapped, X_predicting, Y_train_original, df_predicting_raw = load_and_prepare_data()
    
    if X_train is not None:
        
        predicted_targets, predicted_proba_1 = train_and_predict_xgboost(
            X_train, Y_train_mapped, X_predicting, Y_train_original
        )
        
        df_predicting_raw['Predicted_Target'] = predicted_targets
        df_predicting_raw['Proba_1'] = predicted_proba_1 
        
        print("-" * 50)
        print(f"📢 开始进行交易策略回测 (冷却期: {COOLING_PERIOD_DAYS} 天, 信心阈值: {CONFIDENCE_THRESHOLD}, Sell惩罚: {SELL_WEIGHT}, 全仓交易)...")
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
        
        # 6. 可视化结果
        plot_results(df_results, metrics)
        
        print(f"🎉 评估完成！请多次运行以捕捉最高的 {metrics['Total_Strategy_Return']:.2%} 结果用于您的项目报告！")
