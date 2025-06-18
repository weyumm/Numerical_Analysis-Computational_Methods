#!/usr/bin/env python3
# ------------------------------------------
#  Jacobi 迭代阵 —— 数值实现 (NumPy)
# ------------------------------------------
import numpy as np

def jacobi_iteration_matrix(A: np.ndarray, b=None):
    """
    计算 Jacobi 迭代阵 T 及常向量 c（若给定 b）并返回谱半径 rho(T)。

    Parameters
    ----------
    A : (n,n) ndarray
    b : (n,) ndarray or None   若为 None 仅返回 T 与 rho

    Returns
    -------
    T   : (n,n) ndarray   Jacobi 迭代矩阵
    c   : (n,) ndarray    D^{-1} b  (若 b 为 None 则返回 None)
    rho : float           谱半径
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("A 必须是方阵")

    # D, L, U
    D = np.diag(np.diag(A))
    if np.any(np.diag(D) == 0):
        raise ZeroDivisionError("对角元存在 0，无法构造 D^{-1}")
    D_inv = np.diag(1.0 / np.diag(D))
    T = np.eye(n) - D_inv @ A        # 迭代阵
    eigvals = np.linalg.eigvals(T)
    rho = max(abs(eigvals))

    c = None
    if b is not None:
        b = np.asarray(b, dtype=float).reshape(-1)
        if b.size != n:
            raise ValueError("b 的维度必须与 A 匹配")
        c = D_inv @ b

    return T, c, rho


# --------------------- DEMO ---------------------
if __name__ == "__main__":
    A_demo = np.array([[4, -1, 1],
                       [4, -8, 1],
                       [-2, 1, 5]], dtype=float)
    b_demo = np.array([7, -21, 15], dtype=float)

    T, c, rho = jacobi_iteration_matrix(A_demo, b_demo)
    print("Jacobi 迭代阵 T =\n", T)
    print("常向量 c =", c)
    print("谱半径 ρ(T) =", rho)
