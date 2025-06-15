import numpy as np

A = np.array([[1, 3], [-2, 4]])#修改矩阵
x = np.array([1, -1])

# 计算向量范数
norm_1 = np.linalg.norm(x, 1)
norm_inf = np.linalg.norm(x, np.inf)
norm_2 = np.linalg.norm(x)

# 计算 Ax 的 2-范数
Ax = A.dot(x)
norm_Ax_2 = np.linalg.norm(Ax)

# 计算矩阵范数
matrix_norm_inf = np.linalg.norm(A, np.inf)
matrix_norm_1 = np.linalg.norm(A, 1)

print(f"||x||_1 = {norm_1}")
print(f"||x||_∞ = {norm_inf}")
print(f"||x||_2 = {norm_2:.4f} (即 √{x.dot(x)})")
print(f"||Ax||_2 = {norm_Ax_2:.4f} (即 √{Ax.dot(Ax)} = 2√10)")
print(f"||A||_∞ = {matrix_norm_inf}")
print(f"||A||_1 = {matrix_norm_1}")