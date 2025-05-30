# 计算谱半径，数值
import numpy as np

# 定义矩阵 A
A = np.array([
    [1, 0, 1],
    [2, 2, 1],
    [-1, 0, 0]
], dtype=float)

# 计算特征值
eigenvalues = np.linalg.eigvals(A)

# 谱半径是特征值的最大绝对值
spectral_radius = np.max(np.abs(eigenvalues))

print("特征值:", eigenvalues)
print("谱半径:", spectral_radius)
