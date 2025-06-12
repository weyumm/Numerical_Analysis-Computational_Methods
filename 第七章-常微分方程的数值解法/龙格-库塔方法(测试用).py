def rk2(f, x0, y0, h, xn):
    """
    使用经典龙格-库塔方法求解常微分方程
    
    参数:
    f : 函数 dy/dx = f(x, y)
    x0, y0 : 初始条件
    h : 步长
    xn : 终点
    
    返回:
    x_values : x值列表
    y_values : 对应的y值列表
    """
    x_values = [x0]
    y_values = [y0]
    
    x = x0
    y = y0
    
    # 计算迭代次数
    n = int((xn - x0) / h) + 1
    
    for i in range(1, n):
        k1 = f(x, y)
        k2 = f(x + h, y + k1 * h)
        
        y_next = y + (h/2) * (k1 + k2)
        x_next = x + h
        
        # 更新值
        x = x_next
        y = y_next
        
        x_values.append(x)
        y_values.append(y)
    
    return x_values, y_values

# 定义微分方程
def f(x, y):
    return x + y

# 定义精确解
def exact_solution(x):
    return -x - 1

# 参数设置
x0 = 0.0
y0 = -1.0
h = 0.1
xn = 2.0

# 计算数值解
x_values, y_values = rk2(f, x0, y0, h, xn)

# 计算精确解
y_exact = [exact_solution(x) for x in x_values]

# 计算误差
errors = [abs(y_values[i] - y_exact[i]) for i in range(len(x_values))]

# 输出结果
print("经典R-K方法求解 y' = x+y, y(0)=-1 (h=0.1)")
print("=" * 65)
print("{:<6} {:<12} {:<12} {:<12}".format("x", "RK4", "精确解", "绝对误差"))
print("-" * 65)
for i in range(len(x_values)):
    print("{:<6.1f} {:<12.6f} {:<12.6f} {:<12.6f}".format(
        x_values[i], y_values[i], y_exact[i], errors[i]))
print("=" * 65)

# 输出最大误差
max_error = max(errors)
print(f"最大绝对误差: {max_error:.10f}")