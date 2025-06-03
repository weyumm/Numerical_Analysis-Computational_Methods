import numpy as np

def doolittle_decomposition(A):
    n = len(A)
    L = np.eye(n)  # 初始化 L 为单位下三角矩阵
    U = np.zeros((n, n))  # 初始化 U 为零矩阵

    print("开始杜利特尔分解...\n")

    for i in range(n):
        print(f"===== 第 {i + 1} 步 =====")

        # 第一步：计算 U 的第 i 行
        for k in range(i, n):
            sum_val = sum(L[i][m] * U[m][k] for m in range(i))
            U[i][k] = A[i][k] - sum_val

        # 第二步：计算 L 的第 i 列
        for k in range(i + 1, n):
            sum_val = sum(L[k][m] * U[m][i] for m in range(i))
            if U[i][i] == 0:
                print("无法继续分解：主元为零")
                return None, None
            L[k][i] = (A[k][i] - sum_val) / U[i][i]

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

L, U = doolittle_decomposition(A)

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