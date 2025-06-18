import sympy as sp

def jacobi_analysis(a_val):
    """
    分析 Jacobi 迭代法的收敛性
    
    参数:
    a_val : 符号变量或数值
    
    返回:
    J : Jacobi 迭代矩阵
    rho : 谱半径
    condition : 收敛条件
    """
    # 定义系数矩阵 A
    A = sp.Matrix([
        [a_val, 1, 3],
        [1, a_val, 2],
        [-3, 2, a_val]
    ])
    
    # 分解矩阵 A = D - L - U
    D = sp.diag(A[0, 0], A[1, 1], A[2, 2])
    L = sp.Matrix([
        [0, 0, 0],
        [A[1, 0], 0, 0],
        [A[2, 0], A[2, 1], 0]
    ])
    U = sp.Matrix([
        [0, A[0, 1], A[0, 2]],
        [0, 0, A[1, 2]],
        [0, 0, 0]
    ])
    
    # 计算 Jacobi 迭代矩阵 J = D^{-1}(L + U)
    J = D.inv() * (L + U)
    
    # 计算特征值和谱半径
    eigenvals = J.eigenvals()
    rho = sp.Max(*[abs(val) for val in eigenvals.keys()])
    
    # 收敛条件
    if isinstance(a_val, sp.Symbol):
        condition = sp.solve_univariate_inequality(rho < 1, a_val, relational=False)
    else:
        condition = rho < 1
    
    return A, D, L, U, J, rho, condition

if __name__ == "__main__":
    a = sp.symbols('a')
    
    # 获取分析结果
    A, D, L, U, J, rho, condition = jacobi_analysis(a)
    
    # 打印结果
    print("【问题10】")
    print("\n（1）矩阵分解：")
    print("系数矩阵 A =")
    sp.pretty_print(A)
    print("\n对角矩阵 D =")
    sp.pretty_print(D)
    print("\n下三角矩阵 L =")
    sp.pretty_print(L)
    print("\n上三角矩阵 U =")
    sp.pretty_print(U)
    
    print("\n（2）Jacobi 迭代矩阵 J = D⁻¹(L + U) =")
    sp.pretty_print(J)
    
    print("\n（3）谱半径 ρ(J) =")
    sp.pretty_print(rho)
    
    print("\n（4）收敛条件：")
    sp.pretty_print(condition)