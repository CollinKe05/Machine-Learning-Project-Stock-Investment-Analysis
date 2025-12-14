import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import os

# 修复 Intel MKL 错误
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# --- 配置 ---
TEST_FILE = "data_backtest.csv"
MODEL_FILE = "xgb_model.pkl"
SCALER_FILE = "scaler.pkl"
INITIAL_CAPITAL = 100000.0

# 港股费率
STAMP_DUTY = 0.001      
COMMISSION = 0.00025    
MIN_COMMISSION = 5.0    
PLATFORM_FEE = 15.0     

FEATURES = [
    'Ret_Lag_1', 'Ret_Lag_2', 'Ret_Lag_5', 
    'RSI_6', 'RSI_12', 
    'MACD_Hist', 'Body_Ratio', 'Bias_20', 'ATR'
]

def calculate_cost(amount):
    """计算港股交易成本"""
    stamp = amount * STAMP_DUTY 
    comm = max(amount * COMMISSION, MIN_COMMISSION)
    return comm + stamp + PLATFORM_FEE

def run_backtest():
    print(f"🚀 [Step 4] 开始回测 (2024-01-01 -> 至今)...")
    
    # 1. 加载数据和模型
    df = pd.read_csv(TEST_FILE, index_col='Date', parse_dates=True)
    
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        print(f"❌ 错误: 缺少模型文件 ({MODEL_FILE}) 或 Scaler 文件 ({SCALER_FILE})。请确保您已成功运行 03_model_train.py。")
        return

    bst = joblib.load(MODEL_FILE) 
    scaler = joblib.load(SCALER_FILE)
    
    # 2. 生成预测信号
    X = df[FEATURES]
    X_scaled = scaler.transform(X)
    
    dtest = xgb.DMatrix(X_scaled)
    pred_probs = bst.predict(dtest)
    pred_class_raw = pred_probs.argmax(axis=1)
    
    # 映射回: 0->-1(跌), 1->0(平), 2->1(涨)
    df['Predicted_Signal'] = pd.Series(pred_class_raw, index=df.index).map({0: -1, 1: 0, 2: 1})
    
    # 增加实际信号列 (为了绘图标记)
    THRESHOLD = 0.01 
    df['Actual_Return'] = df['Close'].pct_change().shift(-1).fillna(0)
    df['Actual_Signal'] = 0
    df.loc[df['Actual_Return'] > THRESHOLD, 'Actual_Signal'] = 1
    df.loc[df['Actual_Return'] < -THRESHOLD, 'Actual_Signal'] = -1
    
    # 3. 逐日回测循环
    cash = INITIAL_CAPITAL
    position = 0 
    portfolio_values = []
    trade_log = [] 
    
    # 回测循环：我们遍历到倒数第二天，用今天的信号在明天的开盘价成交
    # i 从 0 到 len(df) - 2
    for i in range(len(df) - 1):
        
        today = df.index[i]
        tomorrow = df.index[i+1]
        
        signal = df['Predicted_Signal'].iloc[i] 
        exec_price = df['Open'].iloc[i+1] # 在第二天的开盘价执行交易
        
        shares_to_trade = 0
        trade_type = "HOLD"
        
        # --- 策略逻辑 V2.0 (稳健策略) ---
        if signal == 1: # 买入
            if position == 0:
                max_val = cash * 0.98
                shares = int(max_val // exec_price)
                shares_to_trade = (shares // 100) * 100 
                
                if shares_to_trade > 0:
                    cost = shares_to_trade * exec_price
                    fee = calculate_cost(cost)
                    
                    if cash >= cost + fee:
                        cash -= (cost + fee)
                        position += shares_to_trade
                        trade_type = "BUY"
                        # 打印交易日志
                        print(f"[{tomorrow.strftime('%Y-%m-%d')}] BUY {shares_to_trade} 股 @ {exec_price:.2f}, 费用: {fee:.2f}, 余额: {cash:.2f}")

        elif signal == -1: # 卖出
            if position > 0:
                revenue = position * exec_price
                fee = calculate_cost(revenue)
                
                cash += (revenue - fee)
                shares_to_trade = position
                position = 0
                trade_type = "SELL"
                # 打印交易日志
                print(f"[{tomorrow.strftime('%Y-%m-%d')}] SELL {shares_to_trade} 股 @ {exec_price:.2f}, 费用: {fee:.2f}, 余额: {cash:.2f}")
            
        # 记录每日资产 (使用今天的收盘价计算持仓市值)
        daily_close = df['Close'].iloc[i]
        total_asset = cash + position * daily_close
        portfolio_values.append(total_asset)
        
        # 记录交易点
        if trade_type == "BUY" or trade_type == "SELL":
            trade_log.append({
                'Date': tomorrow,
                'Type': trade_type,
                'Price': exec_price,
                'Asset': total_asset # 使用交易发生当天的资产净值
            })

    # 补齐最后一天资产值 (使用最后一天的收盘价)
    final_day_close = df['Close'].iloc[-1]
    final_asset = cash + position * final_day_close
    portfolio_values.append(final_asset)
    
    # 4. 结果处理和评估
    
    # 将 Portfolio_Value 赋给 DataFrame (长度现在一致)
    df['Portfolio_Value'] = portfolio_values
    
    # ... (其余评估代码保持不变)
    ret = (final_asset - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    # 简单基准计算
    # 首次买入
    initial_open = df['Open'].iloc[0]
    initial_shares = int((INITIAL_CAPITAL * 0.98) // initial_open)
    initial_cost = initial_shares * initial_open
    buy_fee = calculate_cost(initial_cost)
    
    # 最终卖出
    final_sell_revenue = initial_shares * df['Close'].iloc[-1]
    sell_fee = calculate_cost(final_sell_revenue)
    
    benchmark_final_asset = (initial_shares * df['Close'].iloc[-1]) - sell_fee + (INITIAL_CAPITAL - initial_cost - buy_fee)
    benchmark_ret = (benchmark_final_asset - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    df['Peak'] = df['Portfolio_Value'].cummax()
    df['Drawdown'] = (df['Peak'] - df['Portfolio_Value']) / df['Peak']
    max_dd = df['Drawdown'].max()
    
    # 打印结果
    print("-" * 40)
    print(f"📈 回测结果报告 (V3.0 严重过拟合模型)")
    print(f"初始资金: {INITIAL_CAPITAL:,.2f} CNY")
    print(f"最终资产: {final_asset:,.2f} CNY")
    print(f"策略收益率: {ret:.2%}")
    print(f"基准收益率 (买入并持有): {benchmark_ret:.2%}")
    print(f"超额收益 (Alpha): {ret - benchmark_ret:.2%}")
    print(f"最大回撤: {max_dd:.2%}")
    print("-" * 40)
    
    # --- 5. 画图 (包含交易标记) ---
    
    # 准备绘图数据
    df_trades = pd.DataFrame(trade_log)
    
    plt.figure(figsize=(15, 8))
    
    # 绘制净值曲线
    plt.plot(df.index, df['Portfolio_Value'], label='Strategy AI Net Value', color='red', linewidth=1.5)
    
    # 绘制基准线 
    df['Benchmark_Value'] = (df['Close'] / df['Close'].iloc[0]) * initial_shares * df['Close'].iloc[0] + (INITIAL_CAPITAL - initial_cost - buy_fee)
    plt.plot(df.index, df['Benchmark_Value'], label=f'Benchmark (00700)', color='gray', linestyle='--', linewidth=1)
    
    # 绘制交易标记
    # ... (绘图代码保持不变，请确保您在之前的步骤中复制了完整的绘图代码)
    
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(15, 8))

    # 股票价格图 (显示交易点和净值变化)
    axes[0].plot(df.index, df['Close'], label='Stock Close Price', color='black', alpha=0.7)
    axes[0].set_title('Stock Price and Strategy Trade Signals')
    
    # 在收盘价图上标记买入卖出价
    if not df_trades.empty:
        buy_points = df_trades[df_trades['Type'] == 'BUY']
        sell_points = df_trades[df_trades['Type'] == 'SELL']
        
        axes[0].scatter(buy_points['Date'], buy_points['Price'], marker='^', color='green', s=100, label='Buy Price', zorder=5)
        axes[0].scatter(sell_points['Date'], sell_points['Price'], marker='v', color='red', s=100, label='Sell Price', zorder=5)
    
    axes[0].legend(loc='upper left')
    axes[0].grid(True, axis='y', alpha=0.5)

    # 预测信号 vs 实际信号图
    axes[1].plot(df.index, df['Actual_Signal'], label='Actual Signal (1.0% Threshold)', color='gray', alpha=0.5, drawstyle='steps-post')
    axes[1].plot(df.index, df['Predicted_Signal'], label='Predicted Signal', color='red', alpha=0.7, drawstyle='steps-post')
    axes[1].axhline(y=1, color='green', linestyle=':', linewidth=0.5)
    axes[1].axhline(y=-1, color='red', linestyle=':', linewidth=0.5)
    axes[1].set_yticks([-1, 0, 1])
    axes[1].set_yticklabels(['Sell (-1)', 'Hold (0)', 'Buy (1)'])
    axes[1].set_title('Predicted Signal vs. Actual Signal')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, axis='y', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_backtest()