import sympy as sp

# 定义符号变量和矩阵
l_ambda = sp.symbols('l_ambda')
A = sp.Matrix([
    [1, 0, 1],
    [2, 2, 1],
    [-1, 0, 0]
])

# 计算特征多项式
char_poly = A.charpoly(l_ambda)

# 计算特征值
eigenvalues = A.eigenvals()

# 计算谱半径
spectral_radius = max(abs(ev.evalf()) for ev in eigenvalues)

char_poly_expr = char_poly.as_expr()

print("特征多项式：", char_poly_expr)
print("特征值：", eigenvalues)
print("谱半径：", spectral_radius)
