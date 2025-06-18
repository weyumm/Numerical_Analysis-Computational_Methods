import sympy as sp

# 1. 定义符号
a, b, h, n = sp.symbols('a b h n', real=True)
# 节点 x_n = n*h
x_n = n * h
# 下一节点 x_{n+1}
x_np1 = x_n + h

# 2. 精确解 y_exact(x) = a/2 * x^2 + b*x
y_exact = lambda x: (a/2) * x**2 + b * x

# 3. 取节点上的精确值
y_n_exact = y_exact(x_n)           # y(x_n)
y_np1_exact = y_exact(x_np1)       # y(x_{n+1})

# 4. 右端函数 f(x) = a*x + b
f = lambda x: a * x + b

# 5. 改进欧拉（Heun）法一步：
print("="*50)
print("改进欧拉法 (Heun):")
#   5.1 预测值 y_pred = y_n_exact + h * f(x_n)
y_pred = y_n_exact + h * f(x_n)

#   5.2 校正值 y_heun = y_n_exact + (h/2) * ( f(x_n) + f(x_{n+1}) )
y_heun = y_n_exact + (h/2) * (f(x_n) + f(x_np1))

# 6. 局部截断误差 LTE = 精确下一步 - 数值下一步
tau_heun = sp.simplify(y_np1_exact - y_heun)

# 7. 将结果打印出来
print(f"预测值: {y_pred}")
print(f"校正值: {y_heun}")
print(f"局部截断误差: {tau_heun}")
tau_heun_series = sp.series(tau_heun, h, 0, 4).removeO()
print(f"截断误差展开: {tau_heun_series}")

# 8. 向后欧拉法
print("\n" + "="*50)
print("向后欧拉法:")
# 向后欧拉公式: y_{n+1} = y_n + h * f(x_{n+1})
# 注意: f(x_{n+1}) = a*(x_n + h) + b
y_backward = y_n_exact + h * f(x_np1)

# 局部截断误差
tau_backward = sp.simplify(y_np1_exact - y_backward)

print(f"数值解: {y_backward}")
print(f"局部截断误差: {tau_backward}")
tau_backward_series = sp.series(tau_backward, h, 0, 4).removeO()
print(f"截断误差展开: {tau_backward_series}")

# 9. 中点欧拉法
print("\n" + "="*50)
print("中点欧拉法:")
# 中点欧拉公式:
# 1. 先计算中点斜率: f(x_n + h/2, y(x_n) + (h/2)*f(x_n))
# 2. y_{n+1} = y_n + h * f(x_n + h/2, ...)
# 由于f与y无关: f(x_n + h/2) = a*(x_n + h/2) + b
y_midpoint = y_n_exact + h * f(x_n + h/2)

# 局部截断误差
tau_midpoint = sp.simplify(y_np1_exact - y_midpoint)

print(f"数值解: {y_midpoint}")
print(f"局部截断误差: {tau_midpoint}")
tau_midpoint_series = sp.series(tau_midpoint, h, 0, 4).removeO()
print(f"截断误差展开: {tau_midpoint_series}")

# 10. 结果比较
print("\n" + "="*50)
print("方法比较:")
print(f"精确解: {y_np1_exact}")
print(f"改进欧拉法: {y_heun}")
print(f"向后欧拉法: {y_backward}")
print(f"中点欧拉法: {y_midpoint}")

print("\n局部截断误差主项:")
print(f"改进欧拉法: {tau_heun_series}")
print(f"向后欧拉法: {tau_backward_series}")
print(f"中点欧拉法: {tau_midpoint_series}")