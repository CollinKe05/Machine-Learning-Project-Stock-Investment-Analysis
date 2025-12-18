import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False


def calculate_macd(df, fast=12, slow=26, signal=9):
    exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
    df['DIF'] = exp1 - exp2
    df['DEA'] = df['DIF'].ewm(span=signal, adjust=False).mean()
    df['MACD_Hist'] = (df['DIF'] - df['DEA']) * 2
    return df


def prepare_features(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df = calculate_macd(df)
    df['Momentum'] = df['Close'] - df['Close'].shift(5)
    df.dropna(inplace=True)
    return df


def plot_feature_correlation_matrix(df, feature_cols, target='Close'):
    """绘制特征间相关性矩阵图"""
    analysis_features = [f for f in feature_cols if f != target]

    feature_corr = df[analysis_features].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(feature_corr, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, square=True,
                cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()

    return feature_corr


def plot_feature_target_correlation(df, feature_cols, target='Close'):
    """绘制特征与目标变量相关性图"""
    analysis_features = [f for f in feature_cols if f != target]

    target_correlations = {}
    for feature in analysis_features:
        corr = df[feature].corr(df[target])
        target_correlations[feature] = {
            '相关系数': corr,
            '绝对相关系数': abs(corr),
        }

    target_df = pd.DataFrame.from_dict(target_correlations, orient='index')
    target_df = target_df.sort_values('绝对相关系数', ascending=False)

    plt.figure(figsize=(14, 8))

    colors = []
    for corr in target_df['相关系数']:
        if corr > 0:
            colors.append(plt.cm.Greens(0.3 + 0.7 * min(corr, 0.8)))
        else:
            colors.append(plt.cm.Reds(0.3 + 0.7 * min(abs(corr), 0.8)))

    bars = plt.barh(range(len(target_df)), target_df['绝对相关系数'], color=colors, height=0.6)
    plt.yticks(range(len(target_df)), target_df.index, fontsize=11)
    plt.xlabel('Absolute Correlation Coefficient with Target Variable', fontsize=12)
    plt.title('Feature Prediction Capability Assessment', fontsize=16, pad=20)
    plt.grid(True, alpha=0.3, axis='x')
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    plt.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    for i, (bar, row) in enumerate(zip(bars, target_df.itertuples())):
        plt.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                 f'{row.相关系数:.3f}', ha='left', va='center', fontsize=10)

    plt.tight_layout()
    plt.show()

    return target_df


def feature_redundancy_analysis(df, feature_cols):
    feature_corr = df[feature_cols].corr()

    high_corr_pairs = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            corr_value = abs(feature_corr.iloc[i, j])
            if corr_value > 0.8:
                high_corr_pairs.append((
                    feature_corr.index[i],
                    feature_corr.columns[j],
                    corr_value
                ))

    if high_corr_pairs:
        print("发现高度相关的特征对（可能冗余）：")
        for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True):
            print(f"  {feat1} 与 {feat2}: 相关系数 = {corr:.3f}")
    else:
        print("未发现高度相关的特征对，特征独立性较好。")

    return high_corr_pairs


def main():
    df = pd.read_csv('00700_cleaned.csv')
    df = prepare_features(df)

    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                    'MA5', 'MA20', 'DIF', 'DEA', 'MACD_Hist']

    print("股票预测特征相关性分析")

    print(f"数据集大小: {len(df)} 条记录")
    print(f"特征数量: {len(feature_cols)} 个")
    print(f"时间范围: {df['Date'].min().date()} 至 {df['Date'].max().date()}")

    print("1. 绘制特征间相关性矩阵图")
    feature_matrix = plot_feature_correlation_matrix(df, feature_cols)
    print("\n")
    print("2. 绘制特征与目标变量相关性图")
    corr_results = plot_feature_target_correlation(df, feature_cols)
    print("\n")
    print("3. 特征冗余性分析")
    redundant_pairs = feature_redundancy_analysis(df, feature_cols)
    print("\n")
    print("4. 分析结果总结")

    top_corr = corr_results.head(3).index.tolist()
    print(f"\n与目标变量相关性最高的特征:")
    for i, feature in enumerate(top_corr, 1):
        corr_value = corr_results.loc[feature, '相关系数']
        print(f"  {i}. {feature}: {corr_value:.3f}")

    print(f"\n特征相关性排名:")
    print(corr_results.to_string())

if __name__ == '__main__':
    main()