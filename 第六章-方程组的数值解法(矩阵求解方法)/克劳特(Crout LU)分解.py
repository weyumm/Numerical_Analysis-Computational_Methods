import numpy as np

def crout_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))  # 下三角矩阵
    U = np.eye(n)         # 单位上三角矩阵

    print("开始克劳特分解...\n")

    for j in range(n):  # 按列处理
        print(f"===== 第 {j + 1} 步 =====")

        # 第一步：计算 L 的第 j 列（从 j 到 n-1 行）
        for i in range(j, n):
            sum_val = sum(L[i][k] * U[k][j] for k in range(j))
            L[i][j] = A[i][j] - sum_val

        # 第二步：计算 U 的第 j 行（j+1 到 n-1 列）
        if L[j][j] == 0:
            print("无法继续分解：主元为零")
            return None, None
        for j_col in range(j + 1, n):
            sum_val = sum(L[j][k] * U[k][j_col] for k in range(j))
            U[j][j_col] = (A[j][j_col] - sum_val) / L[j][j]

        print("当前 L 矩阵：")
        print(np.round(L, 4))
        print("\n当前 U 矩阵：")
        print(np.round(U, 4))
        print()

    print("分解完成！")
    return L, U

# 测试用例
A = np.array([
    [6, 2, 2],
    [6, 4, 8],
    [12, 8, 18]
], dtype=float)

L, U = crout_decomposition(A)

print("最终结果：")
print("L 矩阵：")
print(np.round(L, 4))
print("\nU 矩阵：")
print(np.round(U, 4))

# 验证 A = L @ U
print("\n验证 A = L @ U：")
print("L @ U = ")
print(np.round(L @ U, 4))
print("原始矩阵 A：")
print(A)