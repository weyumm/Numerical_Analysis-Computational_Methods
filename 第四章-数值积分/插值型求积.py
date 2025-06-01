import sympy as sp

# 定义符号变量
x, a = sp.symbols('x a')

# 定义插值节点
x0 = -a
x1 = 0
x2 = a
nodes = [x0, x1, x2]

# 构造Lagrange基函数
def lagrange_basis(i, x, nodes):
    xi = nodes[i]
    terms = [(x - nodes[j]) / (xi - nodes[j]) for j in range(len(nodes)) if j != i]
    return sp.simplify(sp.prod(terms))

# 计算权重 A_i = \int_{-a}^{a} L_i(x) dx
A = []
for i in range(3):
    Li = lagrange_basis(i, x, nodes)
    Ai = sp.integrate(Li, (x, -a, a))
    A.append(sp.simplify(Ai))

# 输出权重表达式
for i, Ai in enumerate(A):
    print(f"A_{i} = {Ai}")

# 构造数值积分函数
def quadrature_rule(f_expr, a_val):
    """使用已构造的公式计算积分近似值"""
    f = sp.lambdify(x, f_expr, modules='numpy')
    x_vals = [-a_val, 0, a_val]
    A_vals = [float(Ai.subs(a, a_val)) for Ai in A]
    return sum(A_vals[i] * f(x_vals[i]) for i in range(3))

# 检验代数精度
print("\n检验代数精度：")
a_val = 1  # 可设为任意正数
for degree in range(10):  # 验证0次到10次多项式
    f_expr = x**degree
    exact = float(sp.integrate(f_expr, (x, -a_val, a_val)))
    approx = quadrature_rule(f_expr, a_val)
    print(f"f(x) = x^{degree:<1d} \t Exact = {exact:.6f}, \t Approx = {approx:.6f}, \t Error = {abs(exact - approx):.2e}")
