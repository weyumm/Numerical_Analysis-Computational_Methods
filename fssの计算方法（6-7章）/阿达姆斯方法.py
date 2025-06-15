import numpy as np

# 定义微分方程
def f(x, y):
    return 1 - y

# 精确解
def exact_solution(x):
    return 1 - np.exp(-x)

# 参数设置
h = 0.2
x_points = np.arange(0, 1.0001, h)  # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
n = len(x_points)

# 初始化数组
y_explicit = np.zeros(n)
y_implicit = np.zeros(n)
y_exact = exact_solution(x_points)

# 已知初始条件
y_explicit[0] = 0.0       # y(0)
y_explicit[1] = 0.181     # y(0.2) 给定启动值
y_implicit[0] = 0.0       # y(0)
y_implicit[1] = 0.181     # y(0.2) 相同启动值

# 显式二阶Adams-Bashforth方法
for i in range(1, n-1):
    # 显式公式: y_{n+1} = y_n + h/2 * [3f(x_n,y_n) - f(x_{n-1},y_{n-1})]
    term1 = 3 * f(x_points[i], y_explicit[i])
    term2 = f(x_points[i-1], y_explicit[i-1])
    y_explicit[i+1] = y_explicit[i] + (h/2) * (term1 - term2)

# 隐式二阶Adams-Moulton方法
for i in range(1, n-1):
    # 隐式公式: y_{n+1} = y_n + h/2 * [f(x_{n+1},y_{n+1}) + f(x_n,y_n)]
    # 由于f(x,y)=1-y是线性的，可解出显式表达式:
    # y_{n+1} = [y_n + (h/2)*(f(x_n,y_n) + 1)] / (1 + h/2)
    numerator = y_implicit[i] + (h/2) * (f(x_points[i], y_implicit[i]) + 1)
    denominator = 1 + h/2
    y_implicit[i+1] = numerator / denominator

# 计算误差
error_explicit = np.abs(y_exact - y_explicit)
error_implicit = np.abs(y_exact - y_implicit)

# 输出结果
print("时间步长: ", x_points)
print("\n显式方法 (Adams-Bashforth):")
print("数值解: ", [f"{y:.6f}" for y in y_explicit])
print("误差:    ", [f"{e:.6f}" for e in error_explicit])

print("\n隐式方法 (Adams-Moulton):")
print("数值解: ", [f"{y:.6f}" for y in y_implicit])
print("误差:    ", [f"{e:.6f}" for e in error_implicit])

# 比较最终点的误差
print("\n在 x=1.0 处的比较:")
print(f"精确解:    {y_exact[-1]:.6f}")
print(f"显式方法解: {y_explicit[-1]:.6f}  误差: {error_explicit[-1]:.6f}")
print(f"隐式方法解: {y_implicit[-1]:.6f}  误差: {error_implicit[-1]:.6f}")
print(f"隐式方法比显式方法精度高: {error_explicit[-1] - error_implicit[-1]:.6f}")