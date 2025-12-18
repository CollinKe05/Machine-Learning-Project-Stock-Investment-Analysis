import pandas as pd

file_path = '00700.txt'

#跳过第一行
data = pd.read_csv(file_path, sep='\t', header=1, encoding='ISO-8859-1')
column_names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Amount']
data.columns = column_names

#日期处理
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)
data.drop_duplicates(subset='Date', inplace=True)
data.reset_index(drop=True, inplace=True)

#异常值处理
abnormal_condition = (
    (data['Open'] <= 0) |
    (data['High'] <= 0) |
    (data['Low'] <= 0) |
    (data['Close'] <= 0) |
    (data['Volume'] < 0)
)
abnormal_count = abnormal_condition.sum()
data = data[~abnormal_condition].reset_index(drop=True)
print(f'检测到并删除的异常值行数: {abnormal_count}')

#缺失值处理
before_rows = len(data)
data.dropna(inplace=True)
after_rows = len(data)
missing_row_count = before_rows - after_rows
print(f'检测到并删除的缺失值行数: {missing_row_count}')

# 保存
output_file = '00700_cleaned.csv'
data.to_csv(output_file, index=False)
print(f'清洗后的 CSV 文件已保存至: {output_file}')
print(f'数据时间范围: {data.Date.min().date()} ~ {data.Date.max().date()}')
print(f'最终样本数量: {len(data)}')
