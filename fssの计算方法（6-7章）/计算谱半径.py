import numpy as np

# 定义矩阵
A = np.array([#修改矩阵
    [1, 0, 1],
    [2, 2, 1],
    [-1, 0, 0]
])

# 计算特征值
eigenvalues = np.linalg.eigvals(A)

# 计算谱半径（最大绝对值）
spectral_radius = np.max(np.abs(eigenvalues))

print("特征值：", eigenvalues)
print("谱半径：", spectral_radius)