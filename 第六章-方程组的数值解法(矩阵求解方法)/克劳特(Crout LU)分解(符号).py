import sympy as sp

def crout_decomposition_symbolic_detailed(A):
    """
    对矩阵 A 做符号化 Crout 分解，并打印详细步骤：
    返回 L, U 两个矩阵，使得 A = L * U，U 对角线为 1.
    """
    A = sp.Matrix(A)
    n = A.rows
    L = sp.zeros(n)
    U = sp.eye(n)
    
    print("开始 Crout 分解（符号版）...\n")
    
    for j in range(n):
        print(f"===== 第 {j+1} 步 =====")
        
        # 计算 L 的第 j 列（从 j 到 n-1 行）
        for i in range(j, n):
            sum_val = sum(L[i, k] * U[k, j] for k in range(j))
            L[i, j] = sp.simplify(A[i, j] - sum_val)
        
        # 检查主元
        if L[j, j] == 0:
            print("无法继续分解：主元为零")
            return None, None
        
        # 计算 U 的第 j 行（列 j+1 到 n-1）
        for k in range(j+1, n):
            sum_val = sum(L[j, t] * U[t, k] for t in range(j))
            U[j, k] = sp.simplify((A[j, k] - sum_val) / L[j, j])
        
        print("当前 L 矩阵：")
        sp.pprint(L)
        print("当前 U 矩阵：")
        sp.pprint(U)
        print()
    
    print("分解完成！\n")
    return L, U

# 测试示例：整数矩阵
A = [
    [6,  2,  2],
    [6,  4,  8],
    [12, 8, 18]
]

L, U = crout_decomposition_symbolic_detailed(A)

print("最终结果：")
print("L =")
sp.pprint(L)
print("\nU =")
sp.pprint(U)

print("\n验证 L * U = A：")
sp.pprint(L * U)

# 测试示例（含参数 a 的符号案例）
a = sp.symbols('a')
A_sym = [[1, 2,   a],
         [3, 4,   5],
         [6, 7,   8]]
b_sym = [9, 10, 11]

L_sym, U_sym = crout_decomposition_symbolic_detailed(A_sym)

print("\n符号案例：")
print("L_sym =")
sp.pprint(L_sym)
print("\nU_sym =")
sp.pprint(U_sym)

print("\n验证 L_sym * U_sym = A_sym：")
sp.pprint(L_sym * U_sym)

