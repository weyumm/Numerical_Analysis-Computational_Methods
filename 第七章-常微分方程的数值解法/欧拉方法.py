import sympy as sp

# 定义符号
a, b, h, x, n = sp.symbols('a b h x n')

# 精确解 y(x) = (1/2) a x^2 + b x
y_exact = a/2 * x**2 + b * x

# 欧拉方法：从 x 走到 x + h
y_euler = y_exact + h * (a * x + b)

# 精确解在 x+h 处的值
y_exact_next = y_exact.subs(x, x + h)

# 局部截断误差 (Local Truncation Error, LTE)
tau = sp.simplify(y_exact_next - y_euler)

# 计算通用第 n 步的 y_n（递推和的形式）
k = sp.symbols('k', integer=True, nonnegative=True)
sum_k = sp.summation(k, (k, 0, n - 1))  # Σ_{k=0..n-1} k = n(n-1)/2
y_n = sp.simplify(h * (a * h * sum_k + b * n))

# 计算第 n 步的全局误差：y_exact(x_n) - y_n
x_n = n * h
y_exact_n = y_exact.subs(x, x_n)
error_n = sp.simplify(y_exact_n - y_n)

# 打印结果
print("【局部截断误差 y(xi)】")
print("y(xi) =", tau)

print("\n【欧拉法的符号形式 y_n】")
print("y_n =", y_n)

print("\n【全局误差 e_n = y_exact(x_n) - y_n】")
print("e_n =", error_n)
