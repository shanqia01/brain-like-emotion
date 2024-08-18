import pandas as pd

# 读取数据集
data = pd.read_excel('test2-animation.xlsx', engine='openpyxl')

# 提取目标值列（PPE 列）
target = data['PPE']

# 数据归一化处理，将目标值缩放到 0 到 1 之间
target_normalized = (target - target.min()) / (target.max() - target.min())

# 将归一化后的目标值替换原始数据集中的 PPE 列
data['PPE'] = target_normalized

# 将归一化后的数据写入一个新的 Excel 文件
data.to_excel('PPEGUIYI-animation.xlsx', index=False)

# 打印归一化后的数据集
print(data)
