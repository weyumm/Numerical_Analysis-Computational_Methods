import numpy as np

# 定义被积函数，处理 x = 0 处的可去奇点
def f(x):
    return np.sin(x) / x if x != 0 else 1.0

# 龙贝格积分函数（带过程输出）
def romberg_integration(f, a, b, tol=0.5e-6, max_iter=10):
    R = np.zeros((max_iter, max_iter))

    print("开始龙贝格积分过程...\n")

    # 初始梯形公式
    R[0, 0] = 0.5 * (b - a) * (f(a) + f(b))
    
    print(f"第 0 步 R[0,0] = {R[0, 0]:.8f}")
    print_table(R, 0)

    for n in range(1, max_iter):
        h = (b - a) / (2 ** n)
        sum_f = sum(f(a + (2 * k - 1) * h) for k in range(1, 2 ** (n - 1) + 1))
        R[n, 0] = 0.5 * R[n - 1, 0] + h * sum_f  # 梯形规则细分

        print(f"第 {n} 步 R[{n},0] = {R[n, 0]:.8f}")

        # Richardson 外推
        for m in range(1, n + 1):
            R[n, m] = R[n, m - 1] + (R[n, m - 1] - R[n - 1, m - 1]) / (4 ** m - 1)
            print(f"       R[{n},{m}] = {R[n, m]:.8f}")

        print_table(R, n)

        # 收敛判断
        if abs(R[n, n] - R[n - 1, n - 1]) < tol:
            print(f"\n积分结果为：{R[n, n]:.8f}")
            print(f"使用 {n+1} 次迭代达到精度要求")
            return R[n, n]

    raise ValueError("Romberg 积分在最大迭代次数内未收敛")

# 打印当前 R 表格
def print_table(R, current_row):
    print("当前 Romberg 表：")
    header = [" "] + [f"m={i}" for i in range(current_row+1)]
    print("{:<8}".format(header[0]), end="")
    for col in header[1:]:
        print("{:<12}".format(col), end="")
    print()

    for i in range(current_row+1):
        print("{:<8}".format(f"n={i}"), end="")
        for j in range(i+1):
            val = f"{R[i, j]:.6f}"
            print("{:<12}".format(val), end="")
        print()
    print()

# 调用函数
result = romberg_integration(f, 0, 1)