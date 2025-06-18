import numpy as np

def gauss_seidel(A, b, x0=None, tol=1e-4, max_iter=100, verbose=True):
    """
    高斯-赛德尔迭代法解线性方程组 Ax = b
    
    参数:
    A : 系数矩阵 (n x n)
    b : 右端向量 (n,)
    x0 : 初始向量 (n,) (默认全零向量)
    tol : 收敛阈值（无穷范数）
    max_iter : 最大迭代次数
    verbose : 是否输出中间步骤
    
    返回:
    x : 解向量 (n,)
    """
    n = A.shape[0]
    assert A.shape == (n, n), "A 必须是方阵"
    assert b.shape == (n,), "b 维度必须与 A 匹配"

    # 分解矩阵 A = D - L - U
    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)

    if verbose:
        print("矩阵分解：")
        print("D =\n", D)
        print("L =\n", L)
        print("U =\n", U)

    # 计算迭代矩阵 G = (D - L)^{-1} @ U
    DL_inv = np.linalg.inv(D - L)
    G = DL_inv @ U
    rho_G = np.max(np.abs(np.linalg.eigvals(G)))

    if verbose:
        print("\n迭代矩阵 G = (D - L)^{-1} @ U")
        print("G =\n", G)
        print(f"谱半径 ρ(G) = {rho_G:.6f}")

    if rho_G >= 1:
        raise ValueError("迭代矩阵谱半径 ≥ 1，高斯-赛德尔法不收敛")

    # 初始化
    x = np.zeros(n) if x0 is None else x0.copy()
    f = DL_inv @ b

    if verbose:
        print("\n迭代过程：")
        print(f"x^(0) = {x}")

    for k in range(1, max_iter + 1):
        x_new = G @ x + f

        if verbose:
            print(f"x^{k} = {x_new}")
            print(f"误差 ε_{k} = {np.linalg.norm(x_new - x, np.inf):.6f}")

        if np.linalg.norm(x_new - x, np.inf) < tol:
            return x_new

        x = x_new

    raise ValueError(f"达到最大迭代次数 {max_iter}，未收敛")

# 示例调用
if __name__ == "__main__":
    A = np.array([
        [0.78, -0.02, -0.12, -0.14],
        [-0.02, 0.86, -0.04, -0.06],
        [-0.12, -0.04, 0.72, -0.08],
        [-0.14, -0.06, -0.08, 0.74]
    ], dtype=float)#####系数矩阵改改改

    b = np.array([0.76, 0.08, 1.12, 0.68], dtype=float)
    #####增广结果改改改

    solution = gauss_seidel(A, b, x0=np.zeros(4), tol=1e-4, max_iter=100, verbose=True)
    print("\n最终解：")
    print(f"x₁ = {solution[0]:.5f}")
    print(f"x₂ = {solution[1]:.5f}")
    print(f"x₃ = {solution[2]:.5f}")
    print(f"x₄ = {solution[3]:.5f}")