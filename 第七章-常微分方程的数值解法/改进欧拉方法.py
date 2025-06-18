"""
使用 Sympy 做符号运算，
针对初值问题 y' = a*x + b, y(0)=0，
用改进欧拉（Heun）法推导数值公式并计算局部截断误差。
精确解: y(x) = (a/2)*x^2 + b*x
"""

import sympy as sp

# 1. 定义符号
a,b,h,n=sp.symbols('a b h n',real=True)
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
#   5.1 预测值 y_pred = y_n_exact + h * f(x_n)
y_pred = y_n_exact + h * f(x_n)

#   5.2 校正值 y_heun = y_n_exact + (h/2) * ( f(x_n) + f(x_{n+1}) )
y_heun = y_n_exact + (h/2) * (f(x_n) + f(x_np1))

# 6. 局部截断误差 LTE = 精确下一步 - 数值下一步
tau = sp.simplify(y_np1_exact - y_heun)

# 7. 将结果打印出来
print("=== 精确解 y(x) ===")
print(f"y(x_n)       = {y_n_exact}")
print(f"y(x_(n+1))   = {y_np1_exact}\n")

print("=== Heun（改进欧拉）一步公式 ===")
print(f"预测 y_pred = y_n + h*f(x_n) = {y_pred}")
print(f"校正 y_heun= y_n + h/2*(f(x_n)+f(x_{n+1})) = {y_heun}\n")

print("=== 局部截断误差 LTE ===")
print(f"tau_n = y_exact(x_(n+1)) - y_heun = {tau}")

# 8. 如果要展开成关于 h 的幂级数，可以检查 tau 在 h → 0 下的展开
tau_series = sp.series(tau, h, 0, 4).removeO()
print("\n=== tau 关于 h 的泰勒展开（低阶项） ===")
print(tau_series)
