# 股票投资分析/optimize/01_data_clean.py
import pandas as pd
import os

# --- 配置 ---
INPUT_FILE = "00700(1).txt"
OUTPUT_FILE = "00700_clean.csv"

def clean_data():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误：未找到文件 {INPUT_FILE}")
        return

    print(f"🚀 [Step 1] 开始清洗数据: {INPUT_FILE}")
    
    # 尝试多种编码读取
    encodings = ['gbk', 'gb18030', 'utf-8']
    df = None
    for enc in encodings:
        try:
            # 假设数据是以制表符分隔的文本文件
            df = pd.read_csv(INPUT_FILE, sep='\t', header=0, skiprows=[0], encoding=enc)
            print(f"✅ 成功读取 (编码: {enc})")
            break
        except Exception:
            continue
            
    if df is None:
        print("❌ 错误：无法读取文件，请检查文件格式或编码。")
        return

    # 清理列名（去除空格）
    df.columns = [c.strip() for c in df.columns]
    
    # 重命名为标准英文名
    col_map = {
        '日期': 'Date', '开盘': 'Open', '最高': 'High', '最低': 'Low', 
        '收盘': 'Close', '成交量': 'Volume', '成交额': 'Amount'
    }
    df.rename(columns=col_map, inplace=True)
    
    # 格式转换
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    except Exception as e:
        print(f"❌ 数据格式转换失败: {e}")
        return
        
    # 删除空值和无效数据
    original_len = len(df)
    df.dropna(inplace=True)
    df = df[df['Volume'] > 0].copy()
    
    # 按日期排序
    df.sort_values('Date', inplace=True)
    
    # 去重
    df.drop_duplicates(subset=['Date'], keep='last', inplace=True)
    
    # 保存
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ 清洗完成，已保存至: {OUTPUT_FILE}")
    print(f"📊 数据清洗统计: 原始 {original_len} 条 -> 清洗后 {len(df)} 条")

if __name__ == "__main__":
    clean_data()