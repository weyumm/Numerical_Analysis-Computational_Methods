import numpy as np

def gaussian_elimination_with_pivoting(A, b):
    n = len(b)
    # 构造增广矩阵
    Augmented = np.hstack([A.astype(float), b.reshape(-1, 1)])
    
    print("初始增广矩阵：")
    print(Augmented)
    print("\n开始消元...\n")

    for i in range(n - 1):
        print(f"===== 第 {i+1} 步 =====")

        # 选主元：找出第i列从第i行到底部中绝对值最大的行
        max_row = i + np.argmax(np.abs(Augmented[i:, i]))
        if max_row != i:
            print(f"交换第 {i} 行 和 第 {max_row} 行")
            Augmented[[i, max_row]] = Augmented[[max_row, i]]
            print("交换后增广矩阵：")
            print(Augmented)

        # 归一化主行（可选，这里不强制归一化）
        pivot = Augmented[i, i]
        if abs(pivot) < 1e-10:
            print("主元接近于零，矩阵可能奇异。")
            return None

        # 消元：将当前列下方所有行变为0
        for j in range(i + 1, n):
            factor = Augmented[j, i] / pivot
            Augmented[j] -= factor * Augmented[i]

        print("消元后增广矩阵：")
        print(Augmented)

    # 回代
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Augmented[i, -1] - np.dot(Augmented[i, i+1:n], x[i+1:n])) / Augmented[i, i]

    print("\n最终上三角增广矩阵：")
    print(Augmented)

    print("\n解向量 x：")
    print(x)

    return x

# 测试用例
A = np.array([
    [2, 1, -1],
    [-3, -1, 2],
    [-2, 9, 2]
])

b = np.array([8, -11, -3])

gaussian_elimination_with_pivoting(A, b)