import numpy as np

def gaussian_elimination(A, b):
    n = len(b)
    # 构造增广矩阵
    Augmented = np.hstack([A.astype(float), b.reshape(-1, 1)])
    
    print("初始增广矩阵：")
    print(Augmented)
    print("\n开始消元...\n")

    # 前向消元，构造上三角矩阵
    for i in range(n):
        # 选取主元（这里简单使用当前行，也可以加选主元避免除零）
        pivot = Augmented[i, i]
        if abs(pivot) < 1e-10:
            print("主元接近于零，可能不可逆。")
            return None

        # 归一化当前行
        Augmented[i] = Augmented[i] / pivot

        # 消去当前列下方的所有元素
        for j in range(i+1, n):
            factor = Augmented[j, i]
            Augmented[j] = Augmented[j] - factor * Augmented[i]

        print(f"第 {i+1} 步消元后：")
        print(Augmented)

    # 回代
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = Augmented[i, -1] - np.dot(Augmented[i, i+1:n], x[i+1:n])

    print("\n最终增广矩阵：")
    print(Augmented)

    print("\n解向量 x：")
    print(x)

    return x

# 测试用例
A = np.array([
    [2, 2, 2],
    [3, 2, 4],
    [1, 3, 9]
])

b = np.array([1,1/2,5/2])

gaussian_elimination(A, b)