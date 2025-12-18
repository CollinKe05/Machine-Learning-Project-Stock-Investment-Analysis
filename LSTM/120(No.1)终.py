import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


# 设置全局随机种子
def set_random_seeds(seed=38):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    random.seed(seed)
    np.random.seed(seed)

    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except:
        pass

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU配置错误: {e}")


# 模型评分函数
def calculate_model_score(r2_val, rmse_val, dir_accuracy, weights=(0.3, 0.3, 0.4)):
    # 归一化处理
    norm_r2 = max(0, min(1, r2_val))
    rmse_score = 100 * math.exp(-rmse_val / 100) if rmse_val > 0 else 100
    norm_dir_acc = dir_accuracy / 100

    # 加权得分
    score = (norm_r2 * weights[0] + rmse_score / 100 * weights[1] + norm_dir_acc * weights[2]) * 100

    return score


# 数据划分参数
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


# 数据加载与划分函数
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


# 方向准确率计算函数
def calculate_direction_accuracy(y_true, y_pred, look_ahead=1):
    if len(y_true) != len(y_pred):
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

    true_directions = []
    for i in range(len(y_true) - look_ahead):
        if y_true[i + look_ahead] > y_true[i]:
            true_directions.append(1)
        else:
            true_directions.append(0)

    pred_directions = []
    for i in range(len(y_pred) - look_ahead):
        if y_pred[i + look_ahead] > y_true[i]:
            pred_directions.append(1)
        else:
            pred_directions.append(0)

    correct = sum(1 for t, p in zip(true_directions, pred_directions) if t == p)
    total = len(true_directions)
    accuracy = (correct / total) * 100 if total > 0 else 0

    return accuracy, correct, total


# 训练策略
def train_with_validation():
    set_random_seeds(SEED)

    print(">>> 读取和处理数据...")
    df = process_stock_data(FILE_PATH)

    pre_train_df = df[df['Date'] <= PRE_TRAIN_END].copy()
    valid_df = df[(df['Date'] >= VALID_START) & (df['Date'] <= VALID_END)].copy()
    test_df_raw = df[df['Date'] >= TEST_START].copy()

    print(
        f"预训练集: {len(pre_train_df)} 天 ({pre_train_df['Date'].min().date()} - {pre_train_df['Date'].max().date()})")
    print(f"验证集: {len(valid_df)} 天 ({valid_df['Date'].min().date()} - {valid_df['Date'].max().date()})")
    print(f"测试集: {len(test_df_raw)} 天 ({test_df_raw['Date'].min().date()} - {test_df_raw['Date'].max().date()})")

    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20',
                    'DIF', 'DEA', 'MACD_Hist']
    target_col = ['Close']

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_Y = MinMaxScaler(feature_range=(0, 1))

    scaler_X.fit(pre_train_df[feature_cols])
    scaler_Y.fit(pre_train_df[target_col])

    pre_train_X_scaled = scaler_X.transform(pre_train_df[feature_cols])
    pre_train_Y_scaled = scaler_Y.transform(pre_train_df[target_col])
    pre_train_X, pre_train_Y = create_dataset(pre_train_X_scaled, pre_train_Y_scaled, LOOKBACK)

    full_valid = pd.concat((pre_train_df.iloc[-LOOKBACK:], valid_df))
    valid_X_scaled_long = scaler_X.transform(full_valid[feature_cols])
    valid_X, _ = create_dataset(valid_X_scaled_long, np.zeros(len(valid_X_scaled_long)), LOOKBACK)

    print("\n>>> 开始训练（10个参数组合）...")

    best_model = None
    best_score = -float('inf')
    best_params = {}
    best_valid_direction_accuracy = 0

    results_history = []

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

        tf.keras.backend.clear_session()
        set_random_seeds(SEED + idx)

        model = Sequential()
        model.add(LSTM(units1, return_sequences=True,
                       input_shape=(LOOKBACK, len(feature_cols))))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units2, return_sequences=False))
        model.add(Dropout(dropout_rate))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(loss='mse', optimizer=optimizer)

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


        valid_predict = model.predict(valid_X, verbose=0)
        valid_predict_real = scaler_Y.inverse_transform(valid_predict)

        valid_len = min(len(valid_df), len(valid_predict_real))
        valid_result_df = valid_df.iloc[:valid_len].copy()
        valid_result_df['Predicted'] = valid_predict_real[:valid_len].flatten()

        y_true_val = valid_result_df['Close'].values
        y_pred_val = valid_result_df['Predicted'].values
        r2_val = r2_score(y_true_val, y_pred_val)
        rmse_val = math.sqrt(mean_squared_error(y_true_val, y_pred_val))

        valid_dir_accuracy, valid_correct, valid_total = calculate_direction_accuracy(
            y_true_val, y_pred_val, look_ahead=1
        )

        model_score = calculate_model_score(
            r2_val,
            rmse_val,
            valid_dir_accuracy,
            weights=(0.3, 0.3, 0.4)
        )


        print(f"组合 {idx} 结果:")
        print(f"验证集R2: {r2_val:.4f}, RMSE: {rmse_val:.2f}")
        print(f"验证集方向准确率: {valid_dir_accuracy:.2f}%")
        print(f"模型综合得分: {model_score:.2f}")
        print(f"{'=' * 60}")

        result_info = {
            '组合': idx,
            '描述': description,
            '验证_R2': r2_val,
            '验证_RMSE': rmse_val,
            '验证_方向准确率': valid_dir_accuracy,
            '模型综合得分': model_score,
        }
        results_history.append(result_info)

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
                '验证_R2': r2_val,
                '验证_RMSE': rmse_val,
                '验证_方向准确率': valid_dir_accuracy,
                '模型综合得分': model_score,
            }


    print("参数搜索完成！最佳模型结果:")

    results_df = pd.DataFrame(results_history)
    results_df = results_df.sort_values('模型综合得分', ascending=False)

    print("\n模型综合得分排序")
    print(results_df[['组合', '描述', '验证_R2', '验证_RMSE',
                      '验证_方向准确率', '模型综合得分']].to_string())

    print(f"\n{'=' * 60}")
    print("最佳模型参数")
    print(f"组合: {best_params['组合']} - {best_params['描述']}")
    print(f"LSTM Units: {best_params['units']}")
    print(f"Dropout Rate: {best_params['dropout']}")
    print(f"Learning Rate: {best_params['learning_rate']}")
    print(f"模型综合得分: {best_params['模型综合得分']:.2f}")
    print(f"验证集R2: {best_params['验证_R2']:.4f}")
    print(f"验证集RMSE: {best_params['验证_RMSE']:.2f}")
    print(f"验证集方向准确率: {best_params['验证_方向准确率']:.2f}%")
    print(f"{'=' * 60}")

    if best_model is not None:
        model_filename = f'best_model_combo{best_params["组合"]}_score{best_score:.1f}.keras'
        best_model.save(model_filename)
        print(f"\n最佳模型已保存为: {model_filename}")

    return best_model, results_df


def main():
    set_random_seeds(SEED)

    print("=" * 60)
    print("LSTM模型参数搜索系统")
    print(f"随机种子: {SEED}")
    print("数据划分: 预训练集 | 验证集 | 测试集")
    print(f"参数搜索: 10个精选组合")
    print("模型选择: 基于R2、RMSE和方向准确率的综合得分")
    print("权重分配: R2(30%), RMSE(30%), 方向准确率(40%)")
    print("=" * 60)

    best_model, results_df = train_with_validation()

    summary = f"""
    {'=' * 60}
                    LSTM模型参数搜索报告
    {'=' * 60}
    数据划分:
    - 预训练集: 截止 {PRE_TRAIN_END}
    - 验证集: {VALID_START} 至 {VALID_END}
    - 测试集: {TEST_START} 起

    模型选择标准:
    - 综合得分 = R2×30% + RMSE得分×30% + 方向准确率×40%
    - 基于验证集指标计算，避免过拟合

    最佳组合表现:
    - 模型综合得分: {results_df.iloc[0]['模型综合得分']:.2f}
    - 验证集R2: {results_df.iloc[0]['验证_R2']:.4f}
    - 验证集RMSE: {results_df.iloc[0]['验证_RMSE']:.2f}
    - 验证集方向准确率: {results_df.iloc[0]['验证_方向准确率']:.2f}%

    总体统计:
    - 平均模型综合得分: {results_df['模型综合得分'].mean():.2f}
    - 最高模型综合得分: {results_df['模型综合得分'].max():.2f}
    - 平均验证方向准确率: {results_df['验证_方向准确率'].mean():.2f}%

    输出:
    - best_model_comboX_scoreX.keras: 最佳模型文件
    {'=' * 60}
    """

    print(summary)


if __name__ == '__main__':
    main()