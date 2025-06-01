import numpy as np

def iterative_refinement(A, b, x0=None, tol=1e-10, max_iter=10):
    """
    用迭代改善法求解 Ax = b
    A, b: numpy 数组
    x0: 初始解（用低精度模拟）
    """
    # 设置使用双精度矩阵
    A = A.astype(np.float64)
    b = b.astype(np.float64)
    
    if x0 is None:
        # 初始解：故意使用低精度计算模拟误差
        A_single = A.astype(np.float32)
        b_single = b.astype(np.float32)
        x = np.linalg.solve(A_single, b_single).astype(np.float64)
    else:
        x = x0.astype(np.float64)

    print(f"初始解 x0 = {x}")
    
    for k in range(max_iter):
        r = b - A @ x  # 计算残差
        if np.linalg.norm(r, ord=np.inf) < tol:
            print(f"第{k}次迭代后收敛，残差为 {r}")
            break
        # 解改正方程 A e = r
        e = np.linalg.solve(A, r)
        x = x + e
        print(f"第{k+1}次迭代，改正量 e = {e}，新解 x = {x}")
    
    return x

if __name__ == "__main__":
    A = np.array([[51, 82],
                  [151/3, 81]], dtype=np.float64)
    b = np.array([235, 232], dtype=np.float64)
    
    x = iterative_refinement(A, b)
    print(f"\n最终精确解为: x = {x}")
