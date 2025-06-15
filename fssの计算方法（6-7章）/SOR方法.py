import numpy as np

# 定义系数矩阵和右端向量
A = np.array([#修改矩阵
    [4, -1, 0, -1, 0, 0],
    [-1, 4, -1, 0, -1, 0],
    [0, -1, 4, 0, 0, -1],
    [-1, 0, 0, 4, -1, 0],
    [0, -1, 0, -1, 4, -1],
    [0, 0, -1, 0, -1, 4]
], dtype=float)

b = np.array([2, 3, 2, 2, 1, 2], dtype=float)#修改向量

def sor_solver(A, b, omega, tol=1e-5, max_iter=1000):#修改精度
    n = len(b)
    x = np.zeros(n)
    for it in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            sigma = np.dot(A[i, :i], x_new[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (1 - omega) * x[i] + omega * (b[i] - sigma) / A[i, i]
        
        # 计算残差
        residual = np.max(np.abs(b - np.dot(A, x_new)))
        if residual < tol:
            return x_new, it + 1
        x = x_new
    return x, max_iter

# 分别计算两种omega的情况
for omega in [1.0, 1.1]:#修改omega
    x_sol, iters = sor_solver(A, b, omega)
    print(f"\nω = {omega:.1f}")
    print(f"迭代次数: {iters}")
    print("数值解:", np.round(x_sol, 6))