import sympy as sp

def gaussian_elimination_symbolic_with_pivoting(A, b):
    """
    对矩阵 A 和向量 b (可含符号或数值) 做增广矩阵的符号高斯消元（含主元选取），
    并打印详细的每一步操作过程。
    
    参数:
      A -- 可转为 sympy.Matrix 的二维可迭代对象
      b -- 可转为 sympy.Matrix 的一维可迭代对象
    
    返回:
      solution -- 字典，键为符号变量 x0, x1, ...，值为求解表达式
    """
    # 转换并构造增广矩阵
    M = sp.Matrix(A).row_join(sp.Matrix(b))
    n = M.rows
    # 生成未知量符号 x0, x1, ..., x{n-1}
    vars = sp.symbols(f'x0:{n}')
    
    print("初始增广矩阵：")
    sp.pprint(M)
    print("\n开始消元...\n")
    
    # 前向消元（含主元选取）
    for i in range(n - 1):
        print(f"===== 第 {i+1} 步 =====")
        # 选主元：在第 i 列第 i 行到最后一行中，找到绝对值最大的行
        # 对数值有效，符号情况下会尝试数值化比较
        col_vals = [abs(M[j, i].evalf()) for j in range(i, n)]
        max_row_offset = col_vals.index(max(col_vals))
        max_row = i + max_row_offset
        if max_row != i:
            print(f"交换 第 {i} 行 和 第 {max_row} 行")
            M.row_swap(i, max_row)
            print("交换后增广矩阵：")
            sp.pprint(M)
        
        pivot = M[i, i]
        if pivot == 0:
            print("主元为零，可能需要后续行交换或系统奇异，跳过归一化与消元。")
            continue
        
        # 归一化第 i 行
        M[i, :] = M[i, :] / pivot
        print(f"归一化第 {i} 行 (主元 = {pivot}) 后：")
        sp.pprint(M)
        
        # 消去第 i 列下方元素
        for j in range(i + 1, n):
            factor = M[j, i]
            M[j, :] = M[j, :] - factor * M[i, :]
            print(f"用第 {i} 行消去第 {j} 行 (因子 = {factor}) 后：")
            sp.pprint(M)
        print()
    
    print("前向消元完成，上三角形式：")
    sp.pprint(M)
    
    # 回代
    print("\n开始回代...")
    solution = {}
    for i in reversed(range(n)):
        # 用已知的解构造 RHS
        rhs = M[i, -1] - sum(M[i, j] * solution[vars[j]] for j in range(i+1, n))
        solution[vars[i]] = sp.simplify(rhs / M[i, i])
        print(f"{vars[i]} = {solution[vars[i]]}")
    
    print("\n求解完毕，通解如下：")
    print(solution)
    return solution

# 测试示例（数值案例）
A_num = [[2, 1, -1],
         [-3, -1, 2],
         [-2, 9, 2]]
b_num = [8, -11, -3]
sol_num = gaussian_elimination_symbolic_with_pivoting(A_num, b_num)

# 测试示例（含参数 a 的符号案例）
a = sp.symbols('a')
A_sym = [[1, 2,   a],
         [3, 4,   5],
         [6, 7,   8]]
b_sym = [9, 10, 11]
sol_sym = gaussian_elimination_symbolic_with_pivoting(A_sym, b_sym)

# 示例2：符号运算精确解
A2 = [[1, 2, 3],
      [4, 5, 6],
      [7, 8, 10]]
b2 = [11, 12, 13]
sol2 = gaussian_elimination_symbolic_with_pivoting(A2, b2)
