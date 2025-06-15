import math

def newton_method1():
    xk = 1.6
    tolerance = 1e-5
    max_iter = 100
    for i in range(max_iter):
        f = xk**2 + 10 * math.cos(xk)
        df = 2 * xk - 10 * math.sin(xk)
        if df == 0:
            return None
        xk_new = xk - f / df
        if abs(xk_new - xk) < tolerance:
            print(f"问题（1）近似根：{xk_new:.6f}")
            return xk_new
        xk = xk_new
    return None

def newton_method2():
    xk = 2.0
    tolerance = 1e-5
    max_iter = 100
    for i in range(max_iter):
        f = 1 + math.atan(xk) - xk
        df = (1 / (1 + xk**2)) - 1
        if df == 0:
            return None
        xk_new = xk - f / df
        if abs(xk_new - xk) < tolerance:
            print(f"问题（2）近似根：{xk_new:.6f}")
            return xk_new
        xk = xk_new
    return None

newton_method1()
newton_method2()