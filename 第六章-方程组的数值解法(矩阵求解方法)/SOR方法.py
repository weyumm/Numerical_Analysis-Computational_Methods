import numpy as np

# 定义线性系统 Ax = b
A = np.array([
    [4, -1, 0, -1, 0, 0],
    [-1, 4, -1, 0, -1, 0],
    [0, -1, 4, 0, 0, -1],
    [-1, 0, 0, 4, -1, 0],
    [0, -1, 0, -1, 4, -1],
    [0, 0, -1, 0, -1, 4]
], dtype=float)

b = np.array([2, 3, 2, 2, 1, 2], dtype=float)

def sor(A, b, omega, tol=1e-10, max_iter=10_000):
    """
    使用逐次超松弛法 (SOR) 求解 Ax = b。

    参数
    ----------
    A : ndarray, shape (n, n)
    b : ndarray, shape (n,)
    omega : float
        松弛参数 (omega = 1 → 高斯-赛德尔法)。
    tol : float, 可选
        更新无穷范数的收敛阈值。
    max_iter : int, 可选
        迭代次数的安全上限。

    返回
    -------
    x : ndarray
        近似解。
    k : int
        执行的迭代次数。
    """
    n = len(b)
    x = np.zeros(n)

    for k in range(1, max_iter + 1):
        x_old = x.copy()

        for i in range(n):
            # 使用最新的值计算求和项
            sigma = sum(A[i, j] * (x[j] if j < i else x_old[j])
                        for j in range(n) if j != i)
            x[i] = (1 - omega) * x_old[i] + omega * (b[i] - sigma) / A[i, i]

        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            return x, k

    raise RuntimeError("SOR 在最大迭代次数内未收敛。")

# 对 omega = 1.0 (高斯-赛德尔法) 和 omega = 1.1 运行 SOR
results = {}
for omega in (1.0, 1.1):
    x, k = sor(A, b, omega)
    residual = np.linalg.norm(A @ x - b, ord=np.inf)
    results[omega] = (x, k, residual)

# 显示结果
for omega, (x, k, res) in results.items():
    print(f"omega = {omega:.1f}")
    print(f"收敛所需的迭代次数: {k}")
    print("解向量:")
    for i, xi in enumerate(x, start=1):
        print(f"  x{i} = {xi:.10f}")
    print(f"残差的无穷范数 ‖Ax − b‖_infty = {res:.3e}")
    print("-" * 40)
