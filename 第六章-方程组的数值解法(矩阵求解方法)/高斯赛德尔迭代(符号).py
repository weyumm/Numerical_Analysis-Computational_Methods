"""
这个代码并不好写，我觉得这个符号版怪怪的，但代码能跑
"""

import sympy as sp

def gauss_seidel_symbolic(A, b, x0=None, max_iter=3):
    """
    符号版高斯-赛德尔迭代展示详细过程。
    
    参数:
      A : sympy.Matrix，可含符号
      b : sympy.Matrix，可含符号
      x0: 初始向量 (sympy.Matrix)，若为 None 则自动生成符号 x0_0...x0_{n-1}
      max_iter: 迭代次数
    
    输出:
      每一步的 D, L, U 分解、迭代矩阵 G，向量 f，及每次迭代 x^(k) 的符号表达式
    """
    A = sp.Matrix(A)
    b = sp.Matrix(b)
    n = A.rows

    # 初始化 x^(0)
    if x0 is None:
        x = sp.Matrix([sp.symbols(f'x0_{i}') for i in range(n)])
    else:
        x = sp.Matrix(x0)

    # 矩阵分解 A = D - L - U
    D = sp.diag(*[A[i, i] for i in range(n)])
    L = -sp.Matrix([[A[i, j] if i > j else 0 for j in range(n)] for i in range(n)])
    U = -sp.Matrix([[A[i, j] if i < j else 0 for j in range(n)] for i in range(n)])

    print("=== 矩阵分解 A = D - L - U ===")
    print("D ="); sp.pprint(D)
    print("L ="); sp.pprint(L)
    print("U ="); sp.pprint(U)

    # 迭代矩阵与常数向量
    DL_inv = (D - L).inv()
    G = sp.simplify(DL_inv * U)
    f = sp.simplify(DL_inv * b)

    print("\n=== 迭代矩阵 G 和向量 f ===")
    print("G = (D - L)^(-1) * U ="); sp.pprint(G)
    print("f = (D - L)^(-1) * b ="); sp.pprint(f)

    # 迭代过程
    for k in range(1, max_iter + 1):
        x = sp.simplify(G * x + f)
        print(f"\n=== 第 {k} 次迭代 x^{k} ===")
        sp.pprint(x)

    return x

# 示例: 2x2 符号矩阵
a, b, c, d, e, f = sp.symbols('a b c d e f')
A = sp.Matrix([[a, b],
               [c, d]])
b_vec = sp.Matrix([e, f])

# 运行示例
solution = gauss_seidel_symbolic(A, b_vec, max_iter=3)

# 示例2
A1 = sp.Matrix([[ 4, -1,  1],
                   [ 4, -8,  1],
                   [-2,  1,  5]], dtype=float)
b1 = sp.Matrix([  7, -21, 15], dtype=float)
solution = gauss_seidel_symbolic(A1, b1, max_iter=10)

A = sp.Matrix([
    [sp.Rational(78,100), sp.Rational(-2,100), sp.Rational(-12,100), sp.Rational(-14,100)],
    [sp.Rational(-2,100), sp.Rational(86,100), sp.Rational(-4,100), sp.Rational(-6,100)],
    [sp.Rational(-12,100), sp.Rational(-4,100), sp.Rational(72,100), sp.Rational(-8,100)],
    [sp.Rational(-14,100), sp.Rational(-6,100), sp.Rational(-8,100), sp.Rational(74,100)]
])
b = sp.Matrix([
    sp.Rational(76,100), sp.Rational(8,100), sp.Rational(112,100), sp.Rational(68,100)
])

# 运行示例迭代
solution = gauss_seidel_symbolic(A, b, max_iter=5)