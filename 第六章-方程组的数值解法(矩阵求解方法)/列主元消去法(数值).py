import numpy as np
from scipy.linalg import solve_triangular

def 列主元消去法(A, b, print_steps=False):
    n = A.shape[0]
    assert A.shape == (n, n), "A必须是方阵"
    assert b.shape == (n,), "b的维度必须与A匹配"
    
    # 创建增广矩阵
    Ab = np.hstack([A, b.reshape(-1, 1)])
    P = np.eye(n)
    L = np.eye(n)
    
    steps = []  # 记录每一步的操作
    
    for i in range(n):
        # 选主元
        max_row = i + np.argmax(np.abs(Ab[i:, i]))
        
        if max_row != i:
            # 交换行
            Ab[[i, max_row]] = Ab[[max_row, i]]
            P[[i, max_row]] = P[[max_row, i]]
            if i > 0:
                L[[i, max_row], :i] = L[[max_row, i], :i]
            # 记录操作并输出
            step_desc = f"交换行 r{i+1} <-> r{max_row+1}"
            steps.append(step_desc)
            if print_steps:
                print_step_matrix(step_desc, Ab)
        
        # 消元过程
        for j in range(i+1, n):
            factor = Ab[j, i] / Ab[i, i]
            L[j, i] = factor
            # 执行行操作
            Ab[j, i:] -= factor * Ab[i, i:]
            # 记录操作并输出
            step_desc = f"r{j+1} = r{j+1} - {factor:.4f}r{i+1}"
            steps.append(step_desc)
            if print_steps:
                print_step_matrix(step_desc, Ab)
    
    # 分解结果
    U = np.triu(Ab[:, :-1])
    b_new = Ab[:, -1]
    
    # 回代求解
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b_new[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    
    return x, P, L, U

def print_step_matrix(step, matrix):
    """格式化输出矩阵的中间状态"""
    print(f"\n # {step}")
    print("当前增广矩阵：")
    for row in matrix:
        print("  " + "  ".join(f"{num:8.4f}" for num in row))
    print("-"*40)

# 示例用法
if __name__ == "__main__":
    A = np.array([[0, 3, 4],
                  [1, -1, 1],
                  [2, 1, 2]], dtype=float)
    b = np.array([1, 2, 3], dtype=float)
    
    x, P, L, U = 列主元消去法(A, b, print_steps=True)
    
    print("\n解为：", x)
    print("\n分解结果：")
    print("置换矩阵 P:")
    print(P)
    print("\n下三角矩阵 L:")
    print(L)
    print("\n上三角矩阵 U:")
    print(U)