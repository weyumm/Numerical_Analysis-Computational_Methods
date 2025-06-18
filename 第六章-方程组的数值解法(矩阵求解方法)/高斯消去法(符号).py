import sympy as sp

def gaussian_elimination_symbolic_detailed(A, b):
    """
    对矩阵 A 和向量 b (可含符号) 做手动符号高斯消元，并打印每一步的增广矩阵变化及回代过程。
    返回一个字典，表示解向量的通解。
    """
    A = sp.Matrix(A)
    b = sp.Matrix(b)
    # 构造增广矩阵
    M = A.row_join(b)
    print("初始增广矩阵：")
    sp.pprint(M)
    
    n = M.rows
    # 前向消元
    for i in range(n):
        pivot = M[i, i]
        # 归一化当前行
        M[i, :] = M[i, :] / pivot
        print(f"\n步骤 {i+1}.1: 用主元 {pivot} 归一化第 {i} 行：")
        sp.pprint(M)
        # 消去当前列下方元素
        for j in range(i+1, n):
            factor = M[j, i]
            M[j, :] = M[j, :] - factor * M[i, :]
            print(f"\n步骤 {i+1}.{j-i+1}: 用因子 {factor} 消去第 {j} 行第 {i} 列：")
            sp.pprint(M)
    
    print("\n前向消元后，上三角形式：")
    sp.pprint(M)
    
    # 回代
    x = sp.symbols(f'x0:{n}')
    solution = {}
    for i in reversed(range(n)):
        # 计算当前未知量
        rhs = M[i, -1] - sum(M[i, j]*solution[x[j]] for j in range(i+1, n))
        solution[x[i]] = sp.simplify(rhs)
        print(f"\n回代: 求解 {x[i]} ：{x[i]} = {solution[x[i]]}")
    
    return solution

# 示例：含参数 a 的方程组
a = sp.symbols('a')
A = [[1, 2,   a],
     [3, 4,   5],
     [6, 7,   8]]
b = [9, 10, 11]

sol = gaussian_elimination_symbolic_detailed(A, b)

print("\n最终通解：")
print(sol)

# 示例2：符号运算精确解
A2 = [[1, 2, 3],
      [4, 5, 6],
      [7, 8, 10]]
b2 = [11, 12, 13]

sol2 = gaussian_elimination_symbolic_detailed(A2, b2)

print("\n最终通解2：")
print(sol2)
