import numpy as np

# 定义系数矩阵和右端向量
A = np.array([
    [4, -1, 0, -1, 0, 0],
    [-1, 4, -1, 0, -1, 0],
    [0, -1, 4, 0, 0, -1],
    [-1, 0, 0, 4, -1, 0],
    [0, -1, 0, -1, 4, -1],
    [0, 0, -1, 0, -1, 4]
], dtype=float)

b = np.array([2, 3, 2, 2, 1, 2], dtype=float)

# 雅可比迭代法
def jacobi_solver(A, b, tol=1e-5, max_iter=1000):
    n = len(b)
    x = np.zeros(n)  # 初始解向量
    x_new = np.zeros(n)  # 存储新解
    
    for it in range(max_iter):
        # 遍历每个方程
        for i in range(n):
            # 计算当前方程中除对角元素外的和
            sigma = 0.0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * x[j]
            
            # 计算新解
            x_new[i] = (b[i] - sigma) / A[i, i]
        
        # 计算残差
        residual = np.max(np.abs(b - np.dot(A, x_new)))
        if residual < tol:
            return x_new, it + 1
        
        # 更新解向量
        x = x_new.copy()
    
    return x, max_iter

# 高斯-塞德尔迭代法
def gauss_seidel_solver(A, b, tol=1e-5, max_iter=1000):
    n = len(b)
    x = np.zeros(n)  # 初始解向量
    
    for it in range(max_iter):
        x_old = x.copy()  # 保存上一轮迭代的解
        
        # 遍历每个方程
        for i in range(n):
            # 计算当前方程中除对角元素外的和
            sigma = 0.0
            # 使用已更新的新值（j < i）
            for j in range(i):
                sigma += A[i, j] * x[j]
            # 使用旧值（j > i）
            for j in range(i+1, n):
                sigma += A[i, j] * x_old[j]
            
            # 计算新解
            x[i] = (b[i] - sigma) / A[i, i]
        
        # 计算残差
        residual = np.max(np.abs(b - np.dot(A, x)))
        if residual < tol:
            return x, it + 1
    
    return x, max_iter

# 使用雅可比法求解
print("雅可比迭代法:")
x_jacobi, iters_jacobi = jacobi_solver(A, b)
print(f"迭代次数: {iters_jacobi}")
print("数值解:", np.round(x_jacobi, 6))

# 使用高斯-塞德尔法求解
print("\n高斯-塞德尔迭代法:")
x_gs, iters_gs = gauss_seidel_solver(A, b)
print(f"迭代次数: {iters_gs}")
print("数值解:", np.round(x_gs, 6))

# 计算精确解用于比较
exact_solution = np.linalg.solve(A, b)
print("\n精确解:", np.round(exact_solution, 6))

# 计算两种方法的误差
error_jacobi = np.max(np.abs(x_jacobi - exact_solution))
error_gs = np.max(np.abs(x_gs - exact_solution))
print(f"\n雅可比法最大误差: {error_jacobi:.6e}")
print(f"高斯-塞德尔法最大误差: {error_gs:.6e}")