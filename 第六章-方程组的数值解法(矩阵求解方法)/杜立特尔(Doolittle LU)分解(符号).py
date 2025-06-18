import sympy as sp

def doolittle_decomposition_symbolic(A):
    """
    对矩阵 A (可含符号) 做 Doolittle LU 分解，打印每一步 L、U 矩阵符号形式。
    返回 L、U 两个矩阵。
    """
    A = sp.Matrix(A)
    n = A.rows
    L = sp.eye(n)
    U = sp.zeros(n)

    print("开始杜利特尔分解...\n")
    for i in range(n):
        print(f"===== 第 {i+1} 步 =====\n")
        # 计算 U 的第 i 行
        for k in range(i, n):
            sum_val = sum(L[i, m] * U[m, k] for m in range(i))
            U[i, k] = sp.simplify(A[i, k] - sum_val)
        # 计算 L 的第 i 列
        for k in range(i+1, n):
            sum_val = sum(L[k, m] * U[m, i] for m in range(i))
            if U[i, i] == 0:
                raise ValueError("无法继续分解：主元 U[{0},{0}] 为零".format(i))
            L[k, i] = sp.simplify((A[k, i] - sum_val) / U[i, i])

        print("当前 L 矩阵：")
        sp.pprint(L)
        print("\n当前 U 矩阵：")
        sp.pprint(U)
        print("\n" + "="*40 + "\n")

    print("分解完成！\n")
    return L, U

# 测试用例（数值矩阵）
A = [
    [6, 2, 2],
    [6, 4, 8],
    [12, 8, 18]
]
L, U = doolittle_decomposition_symbolic(A)

print("最终结果：")
print("L 矩阵：")
sp.pprint(L)
print("\nU 矩阵：")
sp.pprint(U)

# 验证 A = L * U
print("\n验证 A = L * U：")
sp.pprint(L * U)
