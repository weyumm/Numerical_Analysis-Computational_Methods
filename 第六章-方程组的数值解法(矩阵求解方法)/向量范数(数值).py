import numpy as np

# 定义矩阵 A 和向量 x
A = np.array([[1, 3],
              [-2, 4]])
x = np.array([[1],
              [-1]])

# 各种范数的计算
x_1_norm = np.linalg.norm(x, 1)
x_inf_norm = np.linalg.norm(x, np.inf)
x_2_norm = np.linalg.norm(x, 2)

Ax = A @ x

Ax_1_norm = np.linalg.norm(Ax, 1)
Ax_inf_norm = np.linalg.norm(Ax, np.inf)
Ax_2_norm = np.linalg.norm(Ax, 2)

# 打印结果
print(f"||x||_1      = {x_1_norm}")
print(f"||x||_infty      = {x_inf_norm}")
print(f"||x||_2      = {x_2_norm}")
print()
print(f"||Ax||_1     = {Ax_1_norm}")
print(f"||Ax||_infty     = {Ax_inf_norm}")
print(f"||Ax||_2     = {Ax_2_norm}")
