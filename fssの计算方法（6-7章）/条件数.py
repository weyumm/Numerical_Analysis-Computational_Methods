import numpy as np

# 矩阵 (1)
A1 = np.array([[1, 2], [1.001, 2.001]])#修改矩阵
norm_A1_inf = np.linalg.norm(A1, ord=np.inf)
A1_inv = np.linalg.inv(A1)
norm_A1_inv_inf = np.linalg.norm(A1_inv, ord=np.inf)
cond_A1_inf = norm_A1_inf * norm_A1_inv_inf

print("(1) 矩阵 A:")
print(A1)
print(f"无穷范数 ||A||_∞ = {norm_A1_inf}")
print("逆矩阵 A^{-1}:")
print(A1_inv)
print(f"无穷范数 ||A^{-1}||_∞ = {norm_A1_inv_inf}")
print(f"条件数 Cond_∞(A) = {cond_A1_inf:.3f}\n")

# 矩阵 (2)
A2 = np.array([[1, 2], [3, 4]])
norm_A2_inf = np.linalg.norm(A2, ord=np.inf)
A2_inv = np.linalg.inv(A2)
norm_A2_inv_inf = np.linalg.norm(A2_inv, ord=np.inf)
cond_A2_inf = norm_A2_inf * norm_A2_inv_inf

print("(2) 矩阵 A:")
print(A2)
print(f"无穷范数 ||A||_∞ = {norm_A2_inf}")
print("逆矩阵 A^{-1}:")
print(A2_inv)
print(f"无穷范数 ||A^{-1}||_∞ = {norm_A2_inv_inf}")
print(f"条件数 Cond_∞(A) = {cond_A2_inf}")