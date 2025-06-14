import sympy as sp

# 定义符号变量
a_sym, b_sym, h_sym, x_sym, n_sym = sp.symbols('a b h x n')
k = sp.symbols('k', integer=True, nonnegative=True)

# 精确解 y(x) = (1/2) a x^2 + b x
y_exact = a_sym/2 * x_sym**2 + b_sym * x_sym

# 欧拉方法：从 x 走到 x + h
y_euler = y_exact + h_sym * (a_sym * x_sym + b_sym)

# 精确解在 x+h 处的值
y_exact_next = y_exact.subs(x_sym, x_sym + h_sym)

# 局部截断误差 (LTE)
tau = sp.simplify(y_exact_next - y_euler)

# 计算第 n 步的 y_n
sum_k = sp.summation(k, (k, 0, n_sym - 1))  # Σ_{k=0}^{n-1} k = n(n-1)/2
y_n = sp.simplify(h_sym * (a_sym * h_sym * sum_k + b_sym * n_sym))

# 计算第 n 步的全局误差
x_n = n_sym * h_sym
y_exact_n = y_exact.subs(x_sym, x_n)
error_n = sp.simplify(y_exact_n - y_n)

# 打印符号结果（可选）
print("\n【欧拉法的符号形式 y_n】")
print("y_n =", y_n)
print("【局部截断误差 tau】")
print("tau =", tau)
print("\n【全局误差 error_n】")
print("error_n =", error_n)

# 输入具体数值进行计算
a_val = 1.0   # 示例值
b_val = 2.0   # 示例值
h_val = 0.1   # 示例值
n_val = 5     # 示例值

# 将数值代入局部截断误差
tau_val = tau.subs({a_sym: a_val, h_sym: h_val})
print(f"\n局部截断误差 (a={a_val}, h={h_val}) = {tau_val.evalf()}")

# 将数值代入全局误差
error_val = error_n.subs({a_sym: a_val, h_sym: h_val, n_sym: n_val})
print(f"全局误差 (a={a_val}, h={h_val}, n={n_val}) = {error_val.evalf()}")

# 可选：计算第 n 步的精确解和数值解
x_n_val = n_val * h_val
y_exact_val = (a_val/2) * x_n_val**2 + b_val * x_n_val
y_n_val = h_val * (a_val * h_val * (n_val*(n_val-1)/2) + b_val * n_val)
print(f"精确解 y({x_n_val}) = {y_exact_val}")
print(f"数值解 y_n = {y_n_val}")