import numpy as np

# 定义矩阵
A1 = np.array([[1, 2],
               [1.001, 2.001]], dtype=float)

A2 = np.array([[1, 2],
               [3, 4]], dtype=float)

def cond_inf(A):
    """
    :param A: 矩阵
    :return: 矩阵的范数、逆矩阵的范数、条件数
    """
    norm_A = np.linalg.norm(A, ord=np.inf)
    inv_A = np.linalg.inv(A)
    norm_inv_A = np.linalg.norm(inv_A, ord=np.inf)
    return norm_A, norm_inv_A, norm_A * norm_inv_A

# 计算并打印结果
for idx, A in enumerate([A1, A2], start=1):
    norm_A, norm_inv_A, cond = cond_inf(A)
    print(f"A{idx}的条件数为：{cond}")
    print(f"A{idx}的范数为：{norm_A}")
    print(f"A{idx}的逆矩阵的范数为：{norm_inv_A}")
    print(f"A{idx}的条件数为：{cond}")