import numpy as np

def cholesky_solve(A, b):
    # Cholesky分解
    L = np.linalg.cholesky(A)
    print("分解后的下三角矩阵L：")
    print(L)
    print()
    
    # 解Ly = b（前向替换）
    n = len(b)
    y = np.zeros(n)
    print("解方程 Ly = b 的步骤：")
    for i in range(n):
        sum_val = 0
        for j in range(i):
            sum_val += L[i,j] * y[j]
        y[i] = (b[i] - sum_val) / L[i,i]
        # 构造方程字符串
        if i == 0:
            equation = f"{L[i,i]}y{i+1} = {b[i]}"
        else:
            terms = [f"{L[i,k]}y{k+1}" for k in range(i)]
            equation = f"{' + '.join(terms)} + {L[i,i]}y{i+1} = {b[i]}"
        print(f"步骤{i+1}: {equation} → y{i+1} = {y[i]:.0f}")
    print("解得 y = ", y.astype(int))
    print()
    
    # 解L^T x = y（回代）
    LT = L.T
    x = np.zeros(n)
    print("解方程 L^T x = y 的步骤：")
    for i in range(n-1, -1, -1):
        sum_val = 0
        for j in range(i+1, n):
            sum_val += LT[i,j] * x[j]
        x[i] = (y[i] - sum_val) / LT[i,i]
        # 构造方程字符串
        if i == n-1:
            equation = f"{LT[i,i]}x{i+1} = {y[i]}"
        else:
            terms = [f"{LT[i,k]}x{k+1}" for k in range(i+1, n)]
            equation = f"{' + '.join(terms)} + {LT[i,i]}x{i+1} = {y[i]}"
        print(f"步骤{n-i}: {equation} → x{i+1} = {x[i]:.0f}")
    print("解得 x = ", x.astype(int))
    return x

# 定义矩阵A和向量b
A = np.array([[4, 2, -2],
              [2, 2, -3],
              [-2, -3, 14]])
b = np.array([10, 5, 4])

print("用平方根法解方程组：")
print("矩阵A:")
print(A)
print("向量b:")
print(b)
print()

x = cholesky_solve(A, b)

print("\n最终解向量 x:")
print(x.astype(int))