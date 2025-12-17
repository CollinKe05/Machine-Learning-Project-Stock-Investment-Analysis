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
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


# ================= 1. 设置全局随机种子 =================
def set_random_seeds(seed=42):
    """设置所有相关随机种子以确保可重复性"""
    # Python和系统环境
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    # Python随机模块
    random.seed(seed)
    np.random.seed(seed)

    # TensorFlow/Keras
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)

    # 尝试启用确定性操作
    try:
        tf.config.experimental.enable_op_determinism()
    except:
        pass  # 旧版本可能不支持

    # 配置GPU（如果可用）
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU配置错误: {e}")


# ================= 2. 数据划分参数 =================
FILE_PATH = '00700_cleaned.csv'
PRE_TRAIN_END = '2023-06-30'  # 预训练集结束
VALID_START = '2023-07-01'  # 验证集开始
VALID_END = '2023-12-31'  # 验证集结束
TEST_START = '2024-01-01'  # 测试集开始
LOOKBACK = 15
EPOCHS = 60
BATCH_SIZE = 16
COMMISSION = 0
SEED = 42  # 固定随机种子


# ================= 3. 数据加载与划分函数 =================
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


def create_dataset(dataset_X, dataset_Y, look_back=1):
    X, Y = [], []
    for i in range(len(dataset_X) - look_back):
        X.append(dataset_X[i:(i + look_back)])
        Y.append(dataset_Y[i + look_back])
    return np.array(X), np.array(Y)


# ================= 4. 改进的训练策略 =================
def train_with_validation():
    """使用验证集选择最佳模型"""
    # 设置全局随机种子
    set_random_seeds(SEED)

    print(">>> 读取和处理数据...")
    df = process_stock_data(FILE_PATH)

    # 数据划分：三阶段
    pre_train_df = df[df['Date'] <= PRE_TRAIN_END].copy()
    valid_df = df[(df['Date'] >= VALID_START) & (df['Date'] <= VALID_END)].copy()
    test_df_raw = df[df['Date'] >= TEST_START].copy()

    print(
        f"预训练集: {len(pre_train_df)} 天 ({pre_train_df['Date'].min().date()} - {pre_train_df['Date'].max().date()})")
    print(f"验证集: {len(valid_df)} 天 ({valid_df['Date'].min().date()} - {valid_df['Date'].max().date()})")
    print(f"测试集: {len(test_df_raw)} 天 ({test_df_raw['Date'].min().date()} - {test_df_raw['Date'].max().date()})")

    # 特征列
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20',
                    'DIF', 'DEA', 'MACD_Hist']
    target_col = ['Close']

    # 创建scaler（仅在预训练集上拟合）
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_Y = MinMaxScaler(feature_range=(0, 1))

    scaler_X.fit(pre_train_df[feature_cols])
    scaler_Y.fit(pre_train_df[target_col])

    # 准备预训练数据
    pre_train_X_scaled = scaler_X.transform(pre_train_df[feature_cols])
    pre_train_Y_scaled = scaler_Y.transform(pre_train_df[target_col])
    pre_train_X, pre_train_Y = create_dataset(pre_train_X_scaled, pre_train_Y_scaled, LOOKBACK)

    # 准备验证数据
    full_valid = pd.concat((pre_train_df.iloc[-LOOKBACK:], valid_df))
    valid_X_scaled_long = scaler_X.transform(full_valid[feature_cols])
    valid_X, _ = create_dataset(valid_X_scaled_long, np.zeros(len(valid_X_scaled_long)), LOOKBACK)

    # ================= 5. 训练阶段 =================
    print("\n>>> 开始训练（10个参数组合）...")

    # 尝试不同的超参数
    best_model = None
    best_valid_return = -float('inf')
    best_params = {}

    # 记录所有参数组合结果
    results_history = []

    # ================= 10个精选参数组合 =================
    # 精心挑选的10个组合，覆盖不同配置
    parameter_combinations = [
        # (units1, units2, dropout, learning_rate, description)
        (128, 64, 0.2, 0.001, "标准中型网络"),
        (64, 32, 0.3, 0.001, "小型保守网络"),
        (256, 128, 0.2, 0.0005, "大型网络低学习率"),
        (128, 128, 0.3, 0.001, "对称网络"),
        (64, 64, 0.4, 0.001, "紧凑高Dropout"),
        (128, 64, 0.3, 0.0005, "中型保守"),
        (256, 256, 0.2, 0.001, "大型对称"),
        (64, 32, 0.2, 0.0005, "小型低学习率"),
        (128, 128, 0.4, 0.0005, "对称高Dropout"),
        (256, 128, 0.3, 0.001, "大型标准"),
    ]

    total_combinations = len(parameter_combinations)

    for idx, (units1, units2, dropout_rate, lr, description) in enumerate(parameter_combinations, 1):
        print(f"\n--- 尝试组合 {idx}/{total_combinations}: {description} ---")
        print(f"LSTM: ({units1}, {units2}), Dropout: {dropout_rate}, LR: {lr}")

        # 清除之前的计算图并重新设置种子
        tf.keras.backend.clear_session()
        set_random_seeds(SEED + idx)  # 微调种子

        # 构建模型
        model = Sequential()
        model.add(LSTM(units1, return_sequences=True,
                       input_shape=(LOOKBACK, len(feature_cols))))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units2, return_sequences=False))
        model.add(Dropout(dropout_rate))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))

        # 使用自定义学习率的优化器
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(loss='mse', optimizer=optimizer)

        # 早停（监控训练损失）
        early_stop = EarlyStopping(
            monitor='loss',
            patience=6,
            restore_best_weights=True
        )

        # 训练
        history = model.fit(
            pre_train_X, pre_train_Y,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
            callbacks=[early_stop]
        )

        # ================= 6. 验证阶段 =================
        print(">>> 在验证集上评估...")

        # 在验证集上预测
        valid_predict = model.predict(valid_X, verbose=0)
        valid_predict_real = scaler_Y.inverse_transform(valid_predict)

        # 创建验证结果DataFrame
        valid_len = min(len(valid_df), len(valid_predict_real))
        valid_result_df = valid_df.iloc[:valid_len].copy()
        valid_result_df['Predicted'] = valid_predict_real[:valid_len].flatten()

        # 在验证集上执行策略
        valid_return = run_strategy_on_data(valid_result_df)

        # 计算预测准确率指标
        y_true_val = valid_result_df['Close'].values
        y_pred_val = valid_result_df['Predicted'].values
        r2_val = r2_score(y_true_val, y_pred_val)
        rmse_val = math.sqrt(mean_squared_error(y_true_val, y_pred_val))

        print(f"验证集收益率: {valid_return * 100:.2f}%")
        print(f"验证集R2: {r2_val:.4f}, RMSE: {rmse_val:.2f}")

        # 保存结果历史
        result_info = {
            '组合': idx,
            '描述': description,
            'units1': units1,
            'units2': units2,
            'dropout': dropout_rate,
            'learning_rate': lr,
            'valid_return': valid_return,
            'r2_score': r2_val,
            'rmse': rmse_val,
            'epochs_trained': len(history.history['loss'])
        }
        results_history.append(result_info)

        # 保存最佳模型
        if valid_return > best_valid_return:
            best_valid_return = valid_return
            best_model = model
            best_params = {
                '组合': idx,
                '描述': description,
                'units': (units1, units2),
                'dropout': dropout_rate,
                'learning_rate': lr,
                'valid_return': valid_return,
                'r2_score': r2_val,
                'rmse': rmse_val
            }
            print(f"🎯 新的最佳模型！验证收益率: {valid_return * 100:.2f}%")

    # ================= 7. 结果分析 =================
    print(f"\n{'=' * 60}")
    print("参数搜索完成！结果分析:")
    print(f"{'=' * 60}")

    # 显示所有结果
    results_df = pd.DataFrame(results_history)
    results_df = results_df.sort_values('valid_return', ascending=False)

    print("\n📊 所有参数组合结果（按收益率排序）:")
    print(results_df[['组合', '描述', 'valid_return', 'r2_score', 'rmse']].to_string())

    print(f"\n{'=' * 60}")
    print("🎯 最佳模型参数:")
    print(f"组合: {best_params['组合']} - {best_params['描述']}")
    print(f"LSTM Units: {best_params['units']}")
    print(f"Dropout Rate: {best_params['dropout']}")
    print(f"Learning Rate: {best_params['learning_rate']}")
    print(f"验证集收益率: {best_params['valid_return'] * 100:.2f}%")
    print(f"验证集R2 Score: {best_params['r2_score']:.4f}")
    print(f"验证集RMSE: {best_params['rmse']:.2f}")
    print(f"{'=' * 60}")

    # 保存结果历史
    results_df.to_csv('top10_parameter_results.csv', index=False, encoding='utf-8')
    print("✅ 参数搜索结果已保存到 top10_parameter_results.csv")

    # ================= 8. 测试阶段（最终评估） =================
    print("\n>>> 在测试集上评估最佳模型...")

    # 准备测试数据
    full_test = pd.concat((df[df['Date'] <= VALID_END].iloc[-LOOKBACK:], test_df_raw))
    test_X_scaled_long = scaler_X.transform(full_test[feature_cols])
    test_X, _ = create_dataset(test_X_scaled_long, np.zeros(len(test_X_scaled_long)), LOOKBACK)

    # 预测
    test_predict = best_model.predict(test_X, verbose=0)
    test_predict_real = scaler_Y.inverse_transform(test_predict)

    # 创建测试结果
    valid_len = min(len(test_df_raw), len(test_predict_real))
    result_df = test_df_raw.iloc[:valid_len].copy()
    result_df['Predicted'] = test_predict_real[:valid_len].flatten()

    # 计算测试集收益率
    test_return = run_strategy_on_data(result_df)

    # 评估指标
    y_true = result_df['Close'].values
    y_pred = result_df['Predicted'].values
    r2 = r2_score(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n{'=' * 60}")
    print("最终测试结果:")
    print(f"R2 Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"测试集收益率: {test_return * 100:.2f}%")
    print(f"验证集收益率: {best_valid_return * 100:.2f}%")
    print(f"{'=' * 60}")

    # 保存最佳模型
    if test_return > -0.1:  # 允许小幅负收益
        model_filename = f'best_model_top10_combo{best_params["组合"]}.keras'
        best_model.save(model_filename)
        print(f"\n✅ 最佳模型已保存为: {model_filename}")

        # 保存参数记录
        params_record = {
            'best_valid_return': float(best_valid_return),
            'test_return': float(test_return),
            'r2_score': float(r2),
            'rmse': float(rmse),
            'best_params': best_params,
            'data_split': {
                'pre_train_end': PRE_TRAIN_END,
                'valid_start': VALID_START,
                'valid_end': VALID_END,
                'test_start': TEST_START
            },
            'random_seed': SEED,
            'feature_cols': feature_cols,
            'lookback': LOOKBACK,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'all_results': results_history
        }

        with open('best_model_params_top10.json', 'w', encoding='utf-8') as f:
            json.dump(params_record, f, indent=2, ensure_ascii=False)

        # 可视化
        create_final_report(result_df, r2, test_return, best_valid_return, results_df)

    return best_model, result_df, test_return, results_df


# ================= 9. 策略执行函数 =================
def run_strategy_on_data(result_df):
    """在给定数据上执行交易策略"""
    cash = 100000
    position = 0
    assets = []

    # 计算运行中的MA5
    result_df['Run_MA5'] = result_df['Close'].rolling(5).mean().fillna(method='bfill')

    for i in range(len(result_df) - 1):
        price = result_df.iloc[i]['Close']
        pred_next = result_df.iloc[i]['Predicted']
        ma5 = result_df.iloc[i]['Run_MA5']
        dif = result_df.iloc[i]['DIF']
        dea = result_df.iloc[i]['DEA']
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

        elif position > 0:
            trend_bad = price < ma5
            macd_bad = dif < dea
            ai_panic = pred_ret < -0.015
            if (trend_bad and macd_bad) or ai_panic:
                cash += position * price
                position = 0

        assets.append(cash + position * price)

    assets.append(cash + position * result_df.iloc[-1]['Close'])
    result_df['Asset'] = assets

    final_return = (assets[-1] - 100000) / 100000
    return final_return


# ================= 10. 结果可视化 =================
def create_final_report(result_df, r2, test_return, valid_return, results_df):
    """创建最终报告图表"""
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))

    # 子图1：价格预测
    axes[0, 0].plot(result_df['Date'], result_df['Close'], label='真实股价', color='blue', linewidth=2)
    axes[0, 0].plot(result_df['Date'], result_df['Predicted'], label='预测股价',
                    color='orange', linestyle='--', alpha=0.8)
    axes[0, 0].set_title(f'测试集预测对比 | R2: {r2:.4f}', fontsize=14, fontproperties='SimSun')
    axes[0, 0].legend(prop={'family': 'SimSun'}, loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylabel('价格', fontproperties='SimSun')

    # 子图2：参数组合收益率分布
    returns = results_df['valid_return'] * 100
    axes[0, 1].bar(range(len(returns)), returns, color=['red' if r == max(returns) else 'skyblue' for r in returns])
    axes[0, 1].axhline(y=returns.mean(), color='green', linestyle='--', label=f'平均: {returns.mean():.1f}%')
    axes[0, 1].set_title('10个参数组合的验证集收益率', fontsize=14, fontproperties='SimSun')
    axes[0, 1].set_xlabel('参数组合编号', fontproperties='SimSun')
    axes[0, 1].set_ylabel('收益率 (%)', fontproperties='SimSun')
    axes[0, 1].legend(prop={'family': 'SimSun'})
    axes[0, 1].grid(True, alpha=0.3)

    # 子图3：策略净值
    benchmark = result_df['Close'] / result_df['Close'].iloc[0]
    strategy = result_df['Asset'] / 100000

    axes[1, 0].plot(result_df['Date'], benchmark,
                    label=f'基准净值 ({benchmark.iloc[-1] * 100 - 100:.1f}%)', color='gray', alpha=0.7)
    axes[1, 0].plot(result_df['Date'], strategy,
                    label=f'策略净值 ({test_return * 100:.1f}%)', color='red', linewidth=2.5)
    axes[1, 0].set_title(f'测试集表现 | 策略收益: {test_return * 100:.2f}% (验证: {valid_return * 100:.2f}%)',
                         fontsize=14, fontproperties='SimSun')
    axes[1, 0].legend(prop={'family': 'SimSun'}, loc='upper left')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylabel('净值', fontproperties='SimSun')

    # 子图4：参数组合R2分布
    r2_scores = results_df['r2_score']
    axes[1, 1].bar(range(len(r2_scores)), r2_scores,
                   color=['red' if r == max(r2_scores) else 'lightgreen' for r in r2_scores])
    axes[1, 1].axhline(y=r2_scores.mean(), color='blue', linestyle='--', label=f'平均: {r2_scores.mean():.4f}')
    axes[1, 1].set_title('参数组合的R2分数分布', fontsize=14, fontproperties='SimSun')
    axes[1, 1].set_xlabel('参数组合编号', fontproperties='SimSun')
    axes[1, 1].set_ylabel('R2分数', fontproperties='SimSun')
    axes[1, 1].legend(prop={'family': 'SimSun'})
    axes[1, 1].grid(True, alpha=0.3)

    # 子图5：回撤分析
    asset_series = pd.Series(result_df['Asset'].values)
    cumulative_max = asset_series.cummax()
    drawdown = (asset_series - cumulative_max) / cumulative_max * 100
    axes[2, 0].fill_between(result_df['Date'], drawdown, 0, color='red', alpha=0.3)
    axes[2, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[2, 0].axhline(y=-10, color='orange', linestyle='--', alpha=0.5)
    axes[2, 0].axhline(y=-20, color='red', linestyle='--', alpha=0.5)
    max_dd = drawdown.min()
    axes[2, 0].set_title(f'回撤分析 | 最大回撤: {max_dd:.1f}%', fontsize=14, fontproperties='SimSun')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_xlabel('日期', fontproperties='SimSun')
    axes[2, 0].set_ylabel('回撤 (%)', fontproperties='SimSun')

    # 子图6：最佳参数组合特征
    best_row = results_df.iloc[0]
    features = ['units1', 'units2', 'dropout', 'learning_rate']
    values = [best_row['units1'], best_row['units2'],
              best_row['dropout'], best_row['learning_rate']]
    labels = [f'LSTM1: {values[0]}', f'LSTM2: {values[1]}',
              f'Dropout: {values[2]}', f'LR: {values[3]}']

    axes[2, 1].barh(range(len(features)), values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[2, 1].set_yticks(range(len(features)))
    axes[2, 1].set_yticklabels(labels, fontproperties='SimSun')
    axes[2, 1].set_title(f'最佳组合参数 (收益率: {best_row["valid_return"] * 100:.1f}%)',
                         fontsize=14, fontproperties='SimSun')
    axes[2, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('Top10_Parameter_Report.png', dpi=150, bbox_inches='tight')
    plt.show()


# ================= 11. 主函数 =================
def main():
    # 设置全局随机种子
    set_random_seeds(SEED)

    print("=" * 60)
    print("精选LSTM模型训练与验证系统")
    print(f"随机种子: {SEED} (确保结果可重复)")
    print("数据划分: 预训练集 | 验证集 | 测试集")
    print(f"参数搜索: 10个精选组合")
    print("=" * 60)

    model, result_df, test_return, results_df = train_with_validation()

    # 生成总结报告
    summary = f"""
    {'=' * 60}
                精选参数组合验证报告
    {'=' * 60}
    数据划分:
    - 预训练集: 截止 {PRE_TRAIN_END}
    - 验证集: {VALID_START} 至 {VALID_END}
    - 测试集: {TEST_START} 起

    参数搜索:
    - 精选10个参数组合，覆盖不同配置
    - 包含小型、中型、大型网络
    - Dropout率: 0.2-0.4
    - 学习率: 0.001, 0.0005

    最佳组合结果:
    - 验证集收益率: {results_df.iloc[0]['valid_return'] * 100:.2f}%
    - 测试集收益率: {test_return * 100:.2f}%
    - 最佳R2分数: {results_df.iloc[0]['r2_score']:.4f}

    输出文件:
    1. top10_parameter_results.csv - 10个组合详细结果
    2. best_model_top10_comboX.keras - 最佳模型
    3. best_model_params_top10.json - 最佳模型参数
    4. Top10_Parameter_Report.png - 综合报告图表

    使用建议:
    1. 如果测试集表现不佳，可尝试调整策略参数
    2. 查看top10_parameter_results.csv选择其他有潜力的组合
    3. 可修改SEED进行多次实验验证稳定性
    {'=' * 60}
    """

    print(summary)

    with open('Top10_Validation_Summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary)


if __name__ == '__main__':
    main()