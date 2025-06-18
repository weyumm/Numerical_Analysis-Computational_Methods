import sympy as sp

def gaussian_elimination_full_pivot_symbolic(A, b):
    """
    对 A x = b 做全主元（行列）符号高斯消元，打印详细过程。
    参数:
      A -- 可转成 sympy.Matrix 的二维可迭代对象（可含符号）
      b -- 可转成 sympy.Matrix 的一维可迭代对象（可含符号）
    返回:
      dict: {x0: expr0, x1: expr1, ...} 按原始变量顺序
    """
    A = sp.Matrix(A)
    b = sp.Matrix(b)
    n = A.rows

    # 增广矩阵
    M = A.row_join(b)
    print("初始增广矩阵：")
    sp.pprint(M)
    print()

    # 记录变量列的原始顺序
    var_order = list(range(n))
    # 生成符号变量 x0...x{n-1}
    xs = sp.symbols(f'x0:{n}')

    # 前向消元
    for i in range(n-1):
        print(f"===== 第 {i+1} 步 =====")

        # 如果主元为 0，就在子矩阵中找一个非零元素做全主元换列换行
        if M[i, i] == 0:
            found = False
            for r in range(i, n):
                for c in range(i, n):
                    if M[r, c] != 0:
                        # 先换行
                        if r != i:
                            print(f"交换第 {i} 行 与 第 {r} 行")
                            M.row_swap(i, r)
                        # 再换列
                        if c != i:
                            print(f"交换第 {i} 列 与 第 {c} 列（变量 x{var_order[i]} ↔ x{var_order[c]}）")
                            M.col_swap(i, c)
                            var_order[i], var_order[c] = var_order[c], var_order[i]
                        found = True
                        break
                if found:
                    break
            if not found:
                raise ValueError(f"第 {i} 步：找不到可用主元，矩阵可能奇异。")

        print("选定主元后增广矩阵：")
        sp.pprint(M)
        print()

        pivot = M[i, i]
        # 归一化当前行
        print(f"第 {i+1} 步.{i+1}：将第 {i} 行除以主元 {pivot}")
        M[i, :] = M[i, :] / pivot
        sp.pprint(M)
        print()

        # 消去下方
        for j in range(i+1, n):
            factor = M[j, i]
            print(f"第 {i+1} 步.{j+1}：用因子 {factor} 消去第 {j} 行第 {i} 列")
            M[j, :] = M[j, :] - factor * M[i, :]
            sp.pprint(M)
            print()

    print("前向消元完成，上三角形式：")
    sp.pprint(M)
    print()

    # 回代
    sol = {}
    for i in reversed(range(n)):
        # M[i, i]*x_i + sum_{j>i} M[i, j]*x_j = M[i, -1]
        expr = M[i, -1] - sum(M[i, j] * sol[xs[j]] for j in range(i+1, n))
        sol[xs[i]] = sp.simplify(expr / M[i, i])
        print(f"回代求 x{i}:  x{i} = {sol[xs[i]]}")

    # 恢复原始变量顺序
    final = {}
    for idx, xi in enumerate(xs):
        final[xs[var_order.index(idx)]] = sol[xi]

    print("\n变量原始顺序：", var_order)
    print("最终解（按 x0, x1, ... 排列）：")
    for xi in sorted(final, key=lambda v: int(str(v)[1:])):
        print(f"  {xi} = {final[xi]}")

    return final

# —— 测试示例 ——  
a = sp.symbols('a')
A = [
    [1,  2,   a],
    [3,  4,   5],
    [6,  7,   8]
]
b = [9, 10, 11]

sol = gaussian_elimination_full_pivot_symbolic(A, b)

# 示例2：符号运算精确解
A2 = [[1, 2, 3],
      [4, 5, 6],
      [7, 8, 10]]
b2 = [11, 12, 13]

sol2 = gaussian_elimination_full_pivot_symbolic(A2, b2)