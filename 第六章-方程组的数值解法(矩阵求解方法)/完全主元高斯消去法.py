import numpy as np

def gaussian_elimination_full_pivot(A, b):
    n = len(b)
    # 构造增广矩阵 (n x n+1)
    Augmented = np.hstack([A.astype(float), b.reshape(-1, 1)])
    
    # 记录原始变量顺序（用于最后输出结果）
    var_order = np.arange(n)

    print("初始增广矩阵：")
    print(Augmented)
    print()

    for i in range(n - 1):
        print(f"===== 第 {i+1} 步消元 =====")

        # 在子矩阵 Augmented[i:, i:] 中寻找最大绝对值元素的位置
        sub_matrix = np.abs(Augmented[i:, i:n])  # 只考虑系数部分
        max_pos = np.unravel_index(np.argmax(sub_matrix, axis=None), sub_matrix.shape)
        max_row = max_pos[0] + i
        max_col = max_pos[1] + i  # 因为是从第 i 列开始的子矩阵

        if Augmented[max_row, max_col] == 0:
            print("矩阵奇异，无法继续求解。")
            return None

        print(f"找到最大主元位于：行 {max_row}, 列 {max_col}")

        # 行交换
        if max_row != i:
            print(f"交换第 {i} 行 和 第 {max_row} 行")
            Augmented[[i, max_row]] = Augmented[[max_row, i]]

        # 列交换
        if max_col != i:
            print(f"交换第 {i} 列 和 第 {max_col} 列（变量交换）")
            Augmented[:, [i, max_col]] = Augmented[:, [max_col, i]]
            var_order[[i, max_col]] = var_order[[max_col, i]]  # 更新变量顺序记录

        print("交换后增广矩阵：")
        print(Augmented)

        pivot = Augmented[i, i]

        # 对当前列下方所有行进行消元
        for j in range(i + 1, n):
            factor = Augmented[j, i] / pivot
            Augmented[j] -= factor * Augmented[i]

        print("消元后增广矩阵：")
        print(Augmented)
        print()

    # 回代求解
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Augmented[i, -1] - np.dot(Augmented[i, i+1:n], x[i+1:n])) / Augmented[i, i]

    # 恢复变量顺序
    final_x = np.zeros(n)
    for i in range(n):
        final_x[var_order[i]] = x[i]

    print("最终上三角增广矩阵：")
    print(Augmented)

    print("\n变量顺序（列交换后）：", var_order)
    print("解向量 x（按原始变量顺序）：")
    print(final_x)

    return final_x

# 测试用例
A = np.array([
    [2, 1, -1],
    [-3, -1, 2],
    [-2, 9, 2]
])

b = np.array([8, -11, -3])

gaussian_elimination_full_pivot(A, b)