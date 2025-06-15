import sympy as sp
from sympy.matrices import Matrix

def symbolic_cholesky(A):
    n = A.rows
    L = Matrix.zeros(n)
    for i in range(n):
        for j in range(i+1):
            if i == j:
                sum_sq = sum(L[i,k]**2 for k in range(j))
                L[i,j] = sp.sqrt(A[i,j] - sum_sq)
            else:
                sum_prod = sum(L[i,k]*L[j,k] for k in range(j))
                L[i,j] = (A[i,j] - sum_prod) / L[j,j]
    return L

def symbolic_forward_substitution(L, b):
    n = L.rows
    y = Matrix.zeros(n,1)
    steps = []
    for i in range(n):
        sum_val = sum(L[i,j]*y[j] for j in range(i))
        y[i] = (b[i] - sum_val) / L[i,i]
        # 生成计算步骤
        terms = [f"{L[i,j]}*y{j+1}" for j in range(i)]
        equation = f"{' + '.join(terms)}" if terms else ""
        equation += f" {L[i,i]}*y{i+1} = {b[i]}"
        steps.append(f"步骤{i+1}: {equation} → y{i+1} = {sp.simplify(y[i])}")
    return y, steps

def symbolic_back_substitution(LT, y):
    n = LT.rows
    x = Matrix.zeros(n,1)
    steps = []
    for i in range(n-1, -1, -1):
        sum_val = sum(LT[i,j]*x[j] for j in range(i+1, n))
        x[i] = (y[i] - sum_val) / LT[i,i]
        # 生成计算步骤
        terms = [f"{LT[i,j]}*x{j+1}" for j in range(i+1, n)]
        equation = f"{' + '.join(terms)}" if terms else ""
        equation += f" {LT[i,i]}*x{i+1} = {y[i]}"
        steps.append(f"步骤{n-i}: {equation} → x{i+1} = {sp.simplify(x[i])}")
    return x, steps

def solve_with_symbolic_computation():
    # 定义符号矩阵和向量
    A = Matrix([[4, 2, -2],
                [2, 2, -3],
                [-2, -3, 14]])
    b = Matrix([10, 5, 4])

    # Cholesky分解
    L = symbolic_cholesky(A)
    print("分解后的下三角矩阵L：")
    sp.pprint(L)
    print()

    # 前向替换求解y
    print("解方程 Ly = b 的步骤：")
    y, y_steps = symbolic_forward_substitution(L, b)
    for step in y_steps:
        print(step)
    print("解得 y = ")
    sp.pprint(y.applyfunc(sp.simplify))
    print()

    # 回代求解x
    LT = L.T
    print("解方程 L^T x = y 的步骤：")
    x, x_steps = symbolic_back_substitution(LT, y)
    for step in x_steps:
        print(step)
    print("解得 x = ")
    sp.pprint(x.applyfunc(sp.simplify))
    return x

# 执行求解过程
solve_with_symbolic_computation()