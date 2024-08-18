import pandas as pd

# 读取Excel文件
df = pd.read_excel("music-test111.xlsx", engine="openpyxl")

# 保留的特征列
feature_column = 'tonnetz_mean'

# 将特征列中的数据转换为正数
df[feature_column] = df[feature_column].abs()

# 保存修改后的DataFrame回Excel文件
df.to_excel("music-test111.xlsx", index=False)
