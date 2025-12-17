import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import os
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import tensorflow as tf

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


# ================= 1. 设置全局随机种子 =================
def set_random_seeds(seed=38):
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


# ================= 新增：模型评分函数 =================
def calculate_model_score(r2_val, rmse_val, dir_accuracy, weights=(0.3, 0.3, 0.4)):
    """
    计算模型综合得分

    参数:
    - r2_val: R2分数（越大越好）
    - rmse_val: RMSE（越小越好）
    - dir_accuracy: 方向准确率（越大越好）
    - weights: (r2权重, rmse权重, 方向准确率权重)

    返回:
    - score: 综合得分（0-100）
    """
    # 归一化处理
    # R2: 理论上范围为[-∞, 1]，但实践中通常>0，我们限制到[0,1]
    norm_r2 = max(0, min(1, r2_val))

    # RMSE: 需要转换为得分，RMSE越小越好
    # 这里使用相对得分，假设RMSE在合理范围内
    # 使用指数衰减函数将RMSE转换为得分
    rmse_score = 100 * math.exp(-rmse_val / 100) if rmse_val > 0 else 100

    # 方向准确率：已经是百分比，归一化到0-1
    norm_dir_acc = dir_accuracy / 100

    # 计算加权得分
    score = (norm_r2 * weights[0] + rmse_score / 100 * weights[1] + norm_dir_acc * weights[2]) * 100

    return score


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
SEED = 38  # 固定随机种子


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


# ================= 新增：方向准确率计算函数 =================
def calculate_direction_accuracy(y_true, y_pred, look_ahead=1):
    """
    计算方向预测准确率

    参数:
    - y_true: 真实价格数组
    - y_pred: 预测价格数组
    - look_ahead: 预测的时间步长（默认1，即预测下一个时间点）

    返回:
    - accuracy: 方向准确率（百分比）
    - correct_predictions: 正确预测的数量
    - total_predictions: 总预测数量
    - up_accuracy: 上涨预测准确率
    - down_accuracy: 下跌预测准确率
    """
    if len(y_true) != len(y_pred):
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

    # 计算真实方向 (1:上涨, 0:下跌)
    true_directions = []
    for i in range(len(y_true) - look_ahead):
        if y_true[i + look_ahead] > y_true[i]:
            true_directions.append(1)  # 上涨
        else:
            true_directions.append(0)  # 下跌或持平

    # 计算预测方向
    pred_directions = []
    for i in range(len(y_pred) - look_ahead):
        if y_pred[i + look_ahead] > y_true[i]:
            pred_directions.append(1)  # 预测上涨
        else:
            pred_directions.append(0)  # 预测下跌或持平

    # 计算总体准确率
    correct = sum(1 for t, p in zip(true_directions, pred_directions) if t == p)
    total = len(true_directions)
    accuracy = (correct / total) * 100 if total > 0 else 0

    # 计算上涨预测准确率
    true_up_indices = [i for i, d in enumerate(true_directions) if d == 1]
    correct_up = sum(1 for i in true_up_indices if pred_directions[i] == 1)
    up_accuracy = (correct_up / len(true_up_indices) * 100) if true_up_indices else 0

    # 计算下跌预测准确率
    true_down_indices = [i for i, d in enumerate(true_directions) if d == 0]
    correct_down = sum(1 for i in true_down_indices if pred_directions[i] == 0)
    down_accuracy = (correct_down / len(true_down_indices) * 100) if true_down_indices else 0

    return accuracy, correct, total, up_accuracy, down_accuracy


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
    best_score = -float('inf')  # 改为使用综合得分
    best_params = {}
    best_valid_direction_accuracy = 0

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

        # 新增：计算验证集方向准确率
        valid_dir_accuracy, valid_correct, valid_total, valid_up_acc, valid_down_acc = calculate_direction_accuracy(
            y_true_val, y_pred_val, look_ahead=1
        )

        # 计算模型综合得分（使用R2、RMSE和方向准确率）
        model_score = calculate_model_score(
            r2_val,
            rmse_val,
            valid_dir_accuracy,
            weights=(0.3, 0.3, 0.4)  # R2权重30%，RMSE权重30%，方向准确率权重40%
        )

        print(f"验证集R2: {r2_val:.4f}, RMSE: {rmse_val:.2f}")
        print(f"验证集方向准确率: {valid_dir_accuracy:.2f}% ({valid_correct}/{valid_total})")
        print(f"验证集上涨准确率: {valid_up_acc:.2f}%, 下跌准确率: {valid_down_acc:.2f}%")
        print(f"验证集收益率: {valid_return * 100:.2f}%")
        print(f"模型综合得分: {model_score:.2f}")

        # 准备测试数据
        full_test = pd.concat((df[df['Date'] <= VALID_END].iloc[-LOOKBACK:], test_df_raw))
        test_X_scaled_long = scaler_X.transform(full_test[feature_cols])
        test_X, _ = create_dataset(test_X_scaled_long, np.zeros(len(test_X_scaled_long)), LOOKBACK)

        # 预测
        test_predict = model.predict(test_X, verbose=0)
        test_predict_real = scaler_Y.inverse_transform(test_predict)

        # 创建测试结果
        test_len = min(len(test_df_raw), len(test_predict_real))
        result_df = test_df_raw.iloc[:test_len].copy()
        result_df['Predicted'] = test_predict_real[:test_len].flatten()

        # 计算测试集收益率
        test_return = run_strategy_on_data(result_df)

        # 评估指标
        y_true = result_df['Close'].values
        y_pred = result_df['Predicted'].values
        r2 = r2_score(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))

        # 新增：计算测试集方向准确率
        test_dir_accuracy, test_correct, test_total, test_up_acc, test_down_acc = calculate_direction_accuracy(
            y_true, y_pred, look_ahead=1
        )

        print(f"\n{'=' * 60}")
        print(f"组合 {idx} 测试结果:")
        print(f"R2 Score: {r2:.4f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"测试集收益率: {test_return * 100:.2f}%")
        print(f"测试集方向准确率: {test_dir_accuracy:.2f}% ({test_correct}/{test_total})")
        print(f"测试集上涨准确率: {test_up_acc:.2f}%, 下跌准确率: {test_down_acc:.2f}%")
        print(f"验证集R2: {r2_val:.4f}, RMSE: {rmse_val:.2f}")
        print(f"验证集方向准确率: {valid_dir_accuracy:.2f}%")
        print(f"模型综合得分: {model_score:.2f}")
        print(f"{'=' * 60}")

        # 计算回撤
        asset_series = pd.Series(result_df['Asset'].values)
        cumulative_max = asset_series.cummax()
        drawdown = (asset_series - cumulative_max) / cumulative_max * 100
        max_dd = drawdown.min()
        print(f'回撤分析 | 最大回撤: {max_dd:.1f}%')

        # 保存结果历史（新增模型得分字段）
        result_info = {
            '组合': idx,
            '描述': description,
            'units1': units1,
            'units2': units2,
            'dropout': dropout_rate,
            'learning_rate': lr,
            'valid_return': valid_return,
            'valid_r2': r2_val,
            'valid_rmse': rmse_val,
            'valid_direction_accuracy': valid_dir_accuracy,
            'valid_up_accuracy': valid_up_acc,
            'valid_down_accuracy': valid_down_acc,
            'model_score': model_score,
            'test_return': test_return,
            'test_r2': r2,
            'test_rmse': rmse,
            'test_direction_accuracy': test_dir_accuracy,
            'test_up_accuracy': test_up_acc,
            'test_down_accuracy': test_down_acc,
            'max_drawdown': max_dd,
            'epochs_trained': len(history.history['loss'])
        }
        results_history.append(result_info)

        # 保存最佳模型（基于模型综合得分）
        if model_score > best_score:
            best_score = model_score
            best_valid_direction_accuracy = valid_dir_accuracy
            best_model = model
            best_params = {
                '组合': idx,
                '描述': description,
                'units': (units1, units2),
                'dropout': dropout_rate,
                'learning_rate': lr,
                'valid_return': valid_return,
                'valid_r2': r2_val,
                'valid_rmse': rmse_val,
                'valid_direction_accuracy': valid_dir_accuracy,
                'model_score': model_score,
                'test_return': test_return,
                'test_r2': r2,
                'test_rmse': rmse,
                'test_direction_accuracy': test_dir_accuracy,
                'r2_score': r2_val,
                'rmse': rmse_val
            }
            print(f"🎯 新的最佳模型！模型综合得分: {model_score:.2f}, 方向准确率: {valid_dir_accuracy:.2f}%")

    # ================= 7. 结果分析 =================
    print(f"\n{'=' * 60}")
    print("参数搜索完成！结果分析:")
    print(f"{'=' * 60}")

    # 显示所有结果
    results_df = pd.DataFrame(results_history)
    results_df = results_df.sort_values('model_score', ascending=False)

    print("\n📊 所有参数组合结果（按模型综合得分排序）:")
    print(results_df[['组合', '描述', 'model_score', 'valid_direction_accuracy',
                      'valid_r2', 'valid_rmse', 'valid_return',
                      'test_direction_accuracy', 'test_return']].to_string())

    print(f"\n{'=' * 60}")
    print("🎯 最佳模型参数（基于综合得分）:")
    print(f"组合: {best_params['组合']} - {best_params['描述']}")
    print(f"LSTM Units: {best_params['units']}")
    print(f"Dropout Rate: {best_params['dropout']}")
    print(f"Learning Rate: {best_params['learning_rate']}")
    print(f"模型综合得分: {best_params['model_score']:.2f}")
    print(f"验证集R2: {best_params['valid_r2']:.4f}")
    print(f"验证集RMSE: {best_params['valid_rmse']:.2f}")
    print(f"验证集方向准确率: {best_params['valid_direction_accuracy']:.2f}%")
    print(f"验证集收益率: {best_params['valid_return'] * 100:.2f}%")
    print(f"测试集R2: {best_params['test_r2']:.4f}")
    print(f"测试集RMSE: {best_params['test_rmse']:.2f}")
    print(f"测试集方向准确率: {best_params['test_direction_accuracy']:.2f}%")
    print(f"测试集收益率: {best_params['test_return'] * 100:.2f}%")
    print(f"{'=' * 60}")

    # 按方向准确率排序
    print("\n📊 所有参数组合结果（按验证方向准确率排序）:")
    dir_acc_sorted = results_df.sort_values('valid_direction_accuracy', ascending=False)
    print(dir_acc_sorted[['组合', '描述', 'valid_direction_accuracy', 'model_score', 'valid_return',
                          'test_direction_accuracy', 'test_return']].to_string())

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
    test_len = min(len(test_df_raw), len(test_predict_real))
    result_df = test_df_raw.iloc[:test_len].copy()
    result_df['Predicted'] = test_predict_real[:test_len].flatten()

    # 计算测试集收益率
    test_return = run_strategy_on_data(result_df)

    # 评估指标
    y_true = result_df['Close'].values
    y_pred = result_df['Predicted'].values
    r2 = r2_score(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))

    # 最终的方向准确率计算
    final_dir_accuracy, final_correct, final_total, final_up_acc, final_down_acc = calculate_direction_accuracy(
        y_true, y_pred, look_ahead=1
    )

    # 计算回撤
    asset_series = pd.Series(result_df['Asset'].values)
    cumulative_max = asset_series.cummax()
    drawdown = (asset_series - cumulative_max) / cumulative_max * 100
    max_dd = drawdown.min()

    print(f"\n{'=' * 60}")
    print("最终测试结果:")
    print(f"R2 Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"测试集收益率: {test_return * 100:.2f}%")
    print(f"测试集方向准确率: {final_dir_accuracy:.2f}% ({final_correct}/{final_total})")
    print(f"测试集上涨准确率: {final_up_acc:.2f}%, 下跌准确率: {final_down_acc:.2f}%")
    print(f"最大回撤: {max_dd:.1f}%")
    print(f"模型综合得分: {best_score:.2f}")
    print(f"验证集方向准确率: {best_valid_direction_accuracy:.2f}%")
    print(f"{'=' * 60}")

    # 保存最佳模型
    if test_return > -0.1:  # 允许小幅负收益
        model_filename = f'best_model_top10_combo{best_params["组合"]}_score{best_score:.1f}.keras'
        best_model.save(model_filename)
        print(f"\n✅ 最佳模型已保存为: {model_filename}")

        # 保存参数记录（新增模型得分）
        params_record = {
            'best_model_score': float(best_score),
            'best_valid_direction_accuracy': float(best_valid_direction_accuracy),
            'best_valid_r2': float(best_params['valid_r2']),
            'best_valid_rmse': float(best_params['valid_rmse']),
            'test_return': float(test_return),
            'test_direction_accuracy': float(final_dir_accuracy),
            'test_up_accuracy': float(final_up_acc),
            'test_down_accuracy': float(final_down_acc),
            'r2_score': float(r2),
            'rmse': float(rmse),
            'max_drawdown': float(max_dd),
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

        # 可视化（更新图表以包含模型得分信息）
        create_final_report(result_df, r2, test_return, best_score,
                            final_dir_accuracy, best_valid_direction_accuracy, results_df)

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


# ================= 10. 结果可视化（更新） =================
def create_final_report(result_df, r2, test_return, best_score,
                        test_dir_accuracy, valid_dir_accuracy, results_df):
    """创建最终报告图表（更新版，包含模型得分）"""
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))

    # 子图1：价格预测
    axes[0, 0].plot(result_df['Date'], result_df['Close'], label='真实股价', color='blue', linewidth=2)
    axes[0, 0].plot(result_df['Date'], result_df['Predicted'], label='预测股价',
                    color='orange', linestyle='--', alpha=0.8)
    axes[0, 0].set_title(f'测试集预测对比 | R2: {r2:.4f}, 方向准确率: {test_dir_accuracy:.1f}%',
                         fontsize=14, fontproperties='SimSun')
    axes[0, 0].legend(prop={'family': 'SimSun'}, loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylabel('价格', fontproperties='SimSun')

    # 子图2：参数组合模型得分分布
    model_scores = results_df['model_score']
    axes[0, 1].bar(range(len(model_scores)), model_scores,
                   color=['red' if s == max(model_scores) else 'skyblue' for s in model_scores])
    axes[0, 1].axhline(y=model_scores.mean(), color='green', linestyle='--',
                       label=f'平均: {model_scores.mean():.1f}')
    axes[0, 1].set_title('10个参数组合的模型综合得分', fontsize=14, fontproperties='SimSun')
    axes[0, 1].set_xlabel('参数组合编号', fontproperties='SimSun')
    axes[0, 1].set_ylabel('模型综合得分', fontproperties='SimSun')
    axes[0, 1].legend(prop={'family': 'SimSun'})
    axes[0, 1].grid(True, alpha=0.3)

    # 子图3：策略净值
    benchmark = result_df['Close'] / result_df['Close'].iloc[0]
    strategy = result_df['Asset'] / 100000

    axes[1, 0].plot(result_df['Date'], benchmark,
                    label=f'基准净值 ({benchmark.iloc[-1] * 100 - 100:.1f}%)', color='gray', alpha=0.7)
    axes[1, 0].plot(result_df['Date'], strategy,
                    label=f'策略净值 ({test_return * 100:.1f}%)', color='red', linewidth=2.5)
    axes[1, 0].set_title(f'测试集表现 | 收益: {test_return * 100:.2f}% (模型得分: {best_score:.1f})',
                         fontsize=14, fontproperties='SimSun')
    axes[1, 0].legend(prop={'family': 'SimSun'}, loc='upper left')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylabel('净值', fontproperties='SimSun')

    # 子图4：模型得分 vs 方向准确率散点图
    scores = results_df['model_score']
    dir_acc = results_df['valid_direction_accuracy']
    colors = ['red' if i == 0 else 'blue' for i in range(len(scores))]
    axes[1, 1].scatter(dir_acc, scores, c=colors, s=100, alpha=0.7)
    axes[1, 1].axhline(y=scores.mean(), color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].axvline(x=dir_acc.mean(), color='gray', linestyle='--', alpha=0.5)

    # 标记最佳组合
    best_idx = scores.idxmax()
    axes[1, 1].scatter(dir_acc[best_idx], scores[best_idx], c='green', s=200, marker='*',
                       label=f'最佳组合 {best_idx + 1}')

    axes[1, 1].set_title('验证集: 模型得分 vs 方向准确率', fontsize=14, fontproperties='SimSun')
    axes[1, 1].set_xlabel('方向准确率 (%)', fontproperties='SimSun')
    axes[1, 1].set_ylabel('模型综合得分', fontproperties='SimSun')
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

    # 子图6：性能指标对比
    metrics = ['模型综合得分', '验证方向准确率', '验证R2', '验证RMSE']
    metric_values = [
        best_score,
        valid_dir_accuracy,
        results_df.iloc[best_idx]['valid_r2'] * 100,  # R2转换为百分比显示
        results_df.iloc[best_idx]['valid_rmse']
    ]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    bars = axes[2, 1].bar(metrics, metric_values, color=colors)
    axes[2, 1].set_title('最佳模型性能指标', fontsize=14, fontproperties='SimSun')
    axes[2, 1].set_ylabel('得分/百分比', fontproperties='SimSun')
    axes[2, 1].grid(True, alpha=0.3, axis='y')

    # 在柱状图上添加数值标签
    for bar, value, metric in zip(bars, metric_values, metrics):
        height = bar.get_height()
        if metric == '模型综合得分':
            label = f'{value:.1f}'
        elif metric == '验证方向准确率':
            label = f'{value:.1f}%'
        elif metric == '验证R2':
            label = f'{value:.1f}%'
        else:  # 验证RMSE
            label = f'{value:.2f}'
        axes[2, 1].text(bar.get_x() + bar.get_width() / 2., height + 1,
                        label, ha='center', va='bottom', fontproperties='SimSun')

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
    print("模型选择: 基于R2、RMSE和方向准确率的综合得分")
    print("权重分配: R2(30%), RMSE(30%), 方向准确率(40%)")
    print("=" * 60)

    model, result_df, test_return, results_df = train_with_validation()

    # 生成总结报告（更新版，包含模型得分）
    summary = f"""
    {'=' * 60}
                精选参数组合验证报告（基于综合得分）
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

    模型选择标准:
    - 综合得分 = R2×30% + RMSE得分×30% + 方向准确率×40%
    - R2: 越大越好，归一化到[0,1]
    - RMSE: 越小越好，使用指数衰减转换为得分
    - 方向准确率: 越大越好，归一化到[0,1]

    最佳组合结果:
    - 模型综合得分: {results_df.iloc[0]['model_score']:.2f}
    - 验证集R2: {results_df.iloc[0]['valid_r2']:.4f}
    - 验证集RMSE: {results_df.iloc[0]['valid_rmse']:.2f}
    - 验证集方向准确率: {results_df.iloc[0]['valid_direction_accuracy']:.2f}%
    - 验证集收益率: {results_df.iloc[0]['valid_return'] * 100:.2f}%
    - 测试集收益率: {test_return * 100:.2f}%
    - 测试集方向准确率: {results_df.iloc[0]['test_direction_accuracy']:.2f}%
    - 测试集R2: {results_df.iloc[0]['test_r2']:.4f}
    - 测试集RMSE: {results_df.iloc[0]['test_rmse']:.2f}

    得分分析:
    - 平均模型综合得分: {results_df['model_score'].mean():.2f}
    - 最高模型综合得分: {results_df['model_score'].max():.2f}
    - 平均验证集方向准确率: {results_df['valid_direction_accuracy'].mean():.2f}%
    - 最高验证集方向准确率: {results_df['valid_direction_accuracy'].max():.2f}%

    输出文件:
    1. top10_parameter_results.csv - 10个组合详细结果（含综合得分）
    2. best_model_top10_comboX_scoreX.keras - 最佳模型
    3. best_model_params_top10.json - 最佳模型参数
    4. Top10_Parameter_Report.png - 综合报告图表（含得分分析）

    使用建议:
    1. 模型综合得分越高表示预测性能越好
    2. 如果测试集表现不佳，可尝试调整权重分配
    3. 查看top10_parameter_results.csv选择其他有潜力的组合
    4. 可修改SEED进行多次实验验证稳定性
    5. 方向准确率 > 55% 通常被认为是有预测能力的模型
    {'=' * 60}
    """

    print(summary)

    with open('Top10_Validation_Summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary)


if __name__ == '__main__':
    main()