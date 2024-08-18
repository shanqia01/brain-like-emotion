import pandas as pd
import numpy as np

# 加载现有数据集
file_path = 'animation-test111.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')

# 每行重复的次数
expansion_factor = 10  # 你想要扩充到7600，所以每行将被复制10次

# 添加随机噪声的函数
def add_noise(value, noise_factor=0.1):
    return value + np.random.uniform(-noise_factor, noise_factor) * value

# 复制带有随机变化的行
expanded_data = pd.DataFrame()
for index, row in df.iterrows():
    for _ in range(expansion_factor):
        new_row = row.copy()
        # 对数值列（不包括非数值列）添加随机噪声
        for col in new_row.index:
            if pd.api.types.is_numeric_dtype(new_row[col]):
                new_row[col] = add_noise(new_row[col])
        expanded_data = expanded_data.append(new_row, ignore_index=True)

# 将扩展后的数据保存到新的Excel文件
expanded_file_path = r'D:\brain-like\cijiyuan\code\参考\common\expanded-animation-data.xlsx'
expanded_data.to_excel(expanded_file_path, index=False)
