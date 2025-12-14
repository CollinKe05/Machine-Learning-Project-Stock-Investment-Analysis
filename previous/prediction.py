import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import Counter
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# --- 1. 配置参数和文件路径 ---
# 假设脚本与CSV文件在同一目录下，或者使用绝对路径
TRAIN_FILE_NAME = "00700_train_data_final.csv"
PREDICTING_FILE_NAME = "00700_predicting_data_final.csv"

# 最终选择的 Top 5 因子
FINAL_FEATURE_SET = [
    'Return_Lag_1', 'Return_Lag_5', 'Return_Lag_2', 
    'Daily_Return', 'Body_Ratio'
]
TARGET_COLUMN = 'Target'

# --- 2. 数据加载和预处理 ---

def load_and_prepare_data():
    """从CSV文件加载训练集和预测集，并进行标签转换。"""
    try:
        # 加载训练集
        df_train = pd.read_csv(TRAIN_FILE_NAME, index_col='Date', parse_dates=True)
        # 加载预测集 (注意：预测集不参与训练，只用于预测)
        df_predicting = pd.read_csv(PREDICTING_FILE_NAME, index_col='Date', parse_dates=True)
        
    except FileNotFoundError as e:
        print(f"❌ 错误：未找到文件。请确保以下文件存在于当前运行目录下：{e.filename}")
        return None, None, None

    # 分割特征 X 和标签 Y
    X_train = df_train[FINAL_FEATURE_SET]
    Y_train = df_train[TARGET_COLUMN]
    X_predicting = df_predicting[FINAL_FEATURE_SET]
    
    # 标签转换：将 {-1, 0, 1} 映射到 XGBoost 期望的 {0, 1, 2}
    # -1 (跌) -> 0
    # 0 (平) -> 1
    # 1 (涨) -> 2
    Y_train_mapped = Y_train.replace({-1: 0, 0: 1, 1: 2})
    
    print("-" * 50)
    print(f"✅ 数据加载成功！")
    print(f"训练集大小: {len(X_train)} 样本。")
    print(f"预测集大小: {len(X_predicting)} 样本。")
    print("-" * 50)
    
    return X_train, Y_train_mapped, X_predicting, Y_train

# --- 3. XGBoost 模型训练与评估 ---

def train_and_evaluate_xgboost(X_train, Y_train_mapped, X_predicting, Y_train_original):
    
    # 1. 特征标准化：仅在训练集上拟合 (fit) StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # 使用训练集的统计量来转换 (transform) 预测集
    X_predicting_scaled = scaler.transform(X_predicting)
    
    # 2. 初始化 XGBoost 分类器
    # num_class=3: 对应标签 0, 1, 2
    xgb_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_jobs=-1,
        seed=42 # 保证结果可复现
    )

    print("📢 开始训练 XGBoost 模型 (仅使用训练集数据)...")
    
    # 3. 模型训练
    xgb_model.fit(X_train_scaled, Y_train_mapped)
    
    # 4. 训练集准确率评估
    Y_train_pred_mapped = xgb_model.predict(X_train_scaled)
    
    # 将训练集预测结果映射回原始标签 (-1, 0, 1)
    Y_train_pred = pd.Series(Y_train_pred_mapped).replace({0: -1, 1: 0, 2: 1})
    Y_train_pred.index = Y_train_original.index # 对齐索引
    
    train_accuracy = accuracy_score(Y_train_original, Y_train_pred)
    train_f1_macro = f1_score(Y_train_original, Y_train_pred, average='macro')
    
    print("-" * 50)
    print("📈 XGBoost 训练集性能评估结果:")
    print(f"使用的特征: {FINAL_FEATURE_SET}")
    print(f"训练集准确率 (Accuracy): {train_accuracy:.4f}")
    print(f"训练集 F1-Macro Score: {train_f1_macro:.4f}")
    print("-" * 50)
    print("📋 训练集分类报告:")
    print(classification_report(Y_train_original, Y_train_pred))
    print("-" * 50)
    
    # 5. 对预测集进行预测 (Prediction)
    Y_predicting_pred_mapped = xgb_model.predict(X_predicting_scaled)
    
    # 将预测结果映射回原始标签 (-1, 0, 1)
    Y_predicting_pred = pd.Series(Y_predicting_pred_mapped).replace({0: -1, 1: 0, 2: 1})
    Y_predicting_pred.index = X_predicting.index # 对齐索引
    Y_predicting_pred.name = 'Predicted_Target'
    
    print("✅ 投资集 (2024-01-01 到 2025-04-24) 预测完成。")
    print(f"预测结果分布: {Counter(Y_predicting_pred)}")
    
    return Y_predicting_pred

# --- 4. 主程序运行 ---

if __name__ == "__main__":
    
    # 1. 加载和准备数据
    X_train, Y_train_mapped, X_predicting, Y_train_original = load_and_prepare_data()
    
    if X_train is not None:
        # 2. 训练并预测
        predicted_targets = train_and_evaluate_xgboost(
            X_train, Y_train_mapped, X_predicting, Y_train_original
        )
        
        # 3. 结果保存 (可选：将预测结果保存以便后续回测)
        predicted_targets.to_csv("predicted_targets_2024_2025.csv", header=True)
        print("\n💾 预测结果已保存至 'predicted_targets_2024_2025.csv' 文件。")
        print("🎉 下一步：使用预测结果进行交易策略回测。")