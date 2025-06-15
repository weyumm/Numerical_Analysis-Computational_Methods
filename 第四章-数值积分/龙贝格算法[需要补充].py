import numpy as np

# 定义被积函数，处理 x = 0 处的可去奇点
def f(x):
    return np.sin(x) / x if x != 0 else 1.0

# 龙贝格积分函数
def romberg_integration(f, a, b, tol=0.5e-6, max_iter=20):
    R = np.zeros((max_iter, max_iter))
    R[0, 0] = 0.5 * (b - a) * (f(a) + f(b))  # 初始梯形积分
    
    for n in range(1, max_iter):
        h = (b - a) / (2 ** n)
        sum_f = sum(f(a + (2 * k - 1) * h) for k in range(1, 2 ** (n - 1) + 1))
        R[n, 0] = 0.5 * R[n - 1, 0] + h * sum_f  # 梯形规则细分

        # 龙贝格外推
        for m in range(1, n + 1):
            R[n, m] = R[n, m - 1] + (R[n, m - 1] - R[n - 1, m - 1]) / (4 ** m - 1)

        # 收敛判断
        if abs(R[n, n] - R[n - 1, n - 1]) < tol:
            print(f"积分结果为：{R[n, n]:.8f}")
            print(f"使用 {n+1} 次迭代达到精度要求")
            return R[n, n]

    raise ValueError("Romberg 积分在最大迭代次数内未收敛")

# 调用函数
result = romberg_integration(f, 0, 1)
