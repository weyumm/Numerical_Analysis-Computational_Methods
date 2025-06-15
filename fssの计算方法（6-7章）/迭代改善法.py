import numpy as np

# 定义矩阵和向量
A = np.array([[51.0, 82.0], [151.0/3, 81.0]])#修改矩阵
b = np.array([235.0, 232.0])

# 计算初始解
x0 = np.linalg.solve(A, b)
print("初始解 x0 =", x0)

# 第一次迭代改善
r0 = b - A @ x0
d0 = np.linalg.solve(A, r0)
x1 = x0 + d0
print("\n第一次迭代:")
print("残差 r0 =", r0)
print("修正量 d0 =", d0)
print("更新解 x1 =", x1)

# 第二次迭代改善
r1 = b - A @ x1
d1 = np.linalg.solve(A, r1) if not np.allclose(r1, 0) else np.zeros_like(x1)
x2 = x1 + d1
print("\n第二次迭代:")
print("残差 r1 =", r1)
print("最终解 x =", x2)

# 验证精确解
exact_solution = np.array([3.0, 1.0])
print("\n验证:")
print("精确解 =", exact_solution)
print("误差 =", np.linalg.norm(x2 - exact_solution))