import sympy as sp

# 定义矩阵 A 和向量 x
# 全部使用符号精确表示，避免浮点数误差
A = sp.Matrix([[1, 3],
               [-2, 4]])
x = sp.Matrix([1, -1])

# 计算向量范数（符号精确）
x_1_norm   = sum(sp.Abs(e) for e in x)
x_inf_norm = max(sp.Abs(e) for e in x)
x_2_norm   = sp.sqrt(sum(e**2 for e in x))

# 计算 Ax
Ax = A * x

# Ax 的范数
Ax_1_norm   = sum(sp.Abs(e) for e in Ax)
Ax_inf_norm = max(sp.Abs(e) for e in Ax)
Ax_2_norm   = sp.sqrt(sum(e**2 for e in Ax))

# 输出结果
print("精确符号结果：")
print(f"\|x\|_1      = {x_1_norm}")
print(f"\|x\|_infty      = {x_inf_norm}")
print(f"\|x\|_2      = {x_2_norm}\n")

print(f"\|Ax\|_1     = {Ax_1_norm}")
print(f"\|Ax\|_infty     = {Ax_inf_norm}")
print(f"\|Ax\|_2     = {Ax_2_norm}")
