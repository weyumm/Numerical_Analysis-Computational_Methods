def newton_nth_root(a, n, x0, tolerance=1e-6, max_iter=100):
    """
    用牛顿法计算 x = √[n]{a} 的近似值
    :param a: 被开方数 (a > 0)
    :param n: 根次数 (n ≥ 1)
    :param x0: 初始猜测值 (x0 > 0)
    :param tolerance: 迭代停止容差
    :param max_iter: 最大迭代次数
    :return: 近似根
    """
    xk = x0
    for _ in range(max_iter):
        xk_new = ((n - 1) * xk + a / (xk ** (n - 1))) / n
        if abs(xk_new - xk) < tolerance:
            return xk_new
        xk = xk_new
    return xk

# 示例：计算 √[3]{8} (n=3, a=8)
result = newton_nth_root(a=8, n=3, x0=2.0)
print(f"√³8 的近似值: {result:.6f}")  # 输出应为 2.0

# 示例：计算 √[5]{32} (n=5, a=32)
result = newton_nth_root(a=32, n=5, x0=2.0)
print(f"√⁵32 的近似值: {result:.6f}")  # 输出应为 2.0