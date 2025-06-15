def euler_method(a, b, h, steps):
    x = 0.0#修改边界条件
    y = 0.0
    results = [(x, y)]
    for _ in range(steps):
        y = y + h * (a * x + b)
        x = round(x + h, 2)  # Avoid floating point error
        results.append((x, y))
    return results

def improved_euler_method(a, b, h, steps):
    x = 0.0
    y = 0.0
    results = [(x, y)]
    for i in range(steps):
        # Predictor step
        y_pred = y + h * (a * x + b)#修改函数
        x_next = x + h
        # Corrector step
        y = y + (h / 2) * ((a * x + b) + (a * x_next + b + 0 * y_pred))
        x = round(x_next, 2)  # Avoid floating point error
        results.append((x, y))
    return results

def exact_solution(a, b, x):
    return (a/2)*x**2 + b*x#修改精确解

# 示例：a=1, b=1
a = 1.0
b = 1.0
h=0.2
steps=5

# 计算各方法结果
euler_results = euler_method(a, b,h,steps)
improved_euler_results = improved_euler_method(a, b,h,steps)

# 输出结果表格
print("x\tExact\t\tEuler\t\tImproved Euler")
for i in range(len(euler_results)):
    x = euler_results[i][0]
    exact = exact_solution(a, b, x)
    euler_y = euler_results[i][1]
    improved_y = improved_euler_results[i][1]
    print(f"{x:.1f}\t{exact:.4f}\t\t{euler_y:.4f}\t\t{improved_y:.4f}")