import sympy as sp

# 定义符号
x, h = sp.symbols('x h', real=True, positive=True)
A0, A1, A2, alpha = sp.symbols('A0 A1 A2 alpha', real=True)

# ---------------------------------------------------------
# 问题 (1) 处理过程
# 目标公式:
#   I = \int_0^2 f(x) dx approx A0 f(0) + A1 f(1) + A2 f(2)
# 要求对 f(x)=1, x, x^2 分别精确

# 设置多项式测试函数
# 当 f(x)=1 时:
eq0 = sp.Eq(A0*1 + A1*1 + A2*1, sp.integrate(1, (x, 0, 2)))  # 积分结果 2

# 当 f(x)=x 时:
# 注意 f(0)=0, f(1)=1, f(2)=2
eq1 = sp.Eq(A0*0 + A1*1 + A2*2, sp.integrate(x, (x, 0, 2)))   # 积分结果 2

# 当 f(x)=x^2 时:
# f(0)=0, f(1)=1, f(2)=4
eq2 = sp.Eq(A0*0**2 + A1*1**2 + A2*2**2, sp.integrate(x**2, (x, 0, 2)))  # 积分结果 8/3

# 显示方程
print("问题 (1) 的方程组：")
sp.pprint(eq0)
sp.pprint(eq1)
sp.pprint(eq2)

# 求解方程组
sol1 = sp.solve([eq0, eq1, eq2], (A0, A1, A2), dict=True)[0]
print("\n问题 (1) 的解：")
sp.pprint(sol1)

# 验证对 f(x)=x^3 的精确性
# f(0)=0, f(1)=1, f(2)=8; 因此右侧为 A0*0 + A1*1 + A2*8 = A1 + 8 A2
approx_x3 = sol1[A1] + 8*sol1[A2]
exact_x3 = sp.integrate(x**3, (x, 0, 2))
print("\n对于 f(x)=x^3:")
print("数值公式得到：", sp.simplify(approx_x3))
print("精确积分结果：", sp.simplify(exact_x3))

# ---------------------------------------------------------
# 问题 (2) 处理过程
# 目标公式:
#   I = \int_0^h f(x) dx approx (h/2)[f(0)+f(h)] + alpha h^2 [f'(0) - f'(h)]
#
# 依次对 f(x)=1, x, x^2, x^3 进行验证，
# 其中前两次自动满足，关键在于 f(x)=x^2 的条件来确定 alpha。

# f(x)=1 时
# f(0)=1, f(h)=1, f'(0)=0, f'(h)=0
eq_const = sp.Eq((h/2)*(1+1) + alpha*h**2*(0-0), sp.integrate(1, (x, 0, h)))
# 等号两边均为 h，这里自动成立。

# f(x)=x 时
# f(0)=0, f(h)=h, f'(0)=1, f'(h)=1
eq_linear = sp.Eq((h/2)*(0+h) + alpha*h**2*(1-1), sp.integrate(x, (x, 0, h)))
# 左右均为 h^2/2

# f(x)=x^2 时
# f(0)=0, f(h)=h^2, f'(0)=0, f'(h)=2h
# 数值公式： (h/2)*h^2 + alpha*h**2*(0-2h) = h^3/2 - 2alpha*h^3
# 精确积分： \int_0^h x^2 dx = h^3/3
eq_quad = sp.Eq((h/2)*h**2 - 2*alpha*h**3, sp.integrate(x**2, (x, 0, h)))

print("\n问题 (2) 的条件（针对 f(x)=x^2）:")
sp.pprint(eq_quad)

# 求解 alpha
sol2 = sp.solve(eq_quad, alpha)
print("\n问题 (2) 中 alpha 的解：")
sp.pprint(sol2)

# 进一步验证 f(x)=x^3 的情况
# f(x)=x^3 时: f(0)=0, f(h)=h^3, f'(0)=0, f'(h)=3h^2
# 数值公式： (h/2)*h^3 + alpha * h^2*(0-3h^2) = h^4/2 - 3alpha*h^4
approx_x3_q2 = sp.simplify(h**4/2 - 3*sol2[0]*h**4)
exact_x3_q2 = sp.integrate(x**3, (x, 0, h))
print("\n对于 f(x)=x^3 (问题 (2)):")
print("数值公式得到：", sp.simplify(approx_x3_q2))
print("精确积分结果：", sp.simplify(exact_x3_q2))
