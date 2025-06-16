from fractions import Fraction
import numpy as np

def 列主元消去法符号版(A, b, print_steps=False):
    # 将输入列表转换为NumPy数组（保持分数类型）
    A = np.array(A, dtype=object)
    b = np.array(b, dtype=object)
    
    n = A.shape[0]  # 现在可以安全使用shape属性
    assert A.shape == (n, n), "A必须是方阵"
    assert len(b) == n, "b的维度必须与A匹配"
    
    # 转换为分数类型
    A = [[Fraction(x).limit_denominator() for x in row] for row in A]
    b = [Fraction(x).limit_denominator() for x in b]
    
    # 创建增广矩阵
    Ab = [row + [b[i]] for i, row in enumerate(A)]
    P = np.eye(n, dtype=int)
    L = np.eye(n, dtype=object)
    
    steps = []
    
    for i in range(n):
        # 选主元（当前列绝对值最大）
        current_col = [abs(Ab[j][i]) for j in range(i, n)]
        max_idx = i + current_col.index(max(current_col))
        
        if max_idx != i:
            # 交换行
            Ab[i], Ab[max_idx] = Ab[max_idx], Ab[i]
            P[[i, max_idx]] = P[[max_idx, i]]
            if i > 0:
                L[i, :i], L[max_idx, :i] = L[max_idx, :i], L[i, :i]
            steps.append(f"交换行 r{i+1} <-> r{max_idx+1}")
        
        # 消元过程
        pivot = Ab[i][i]
        for j in range(i+1, n):
            factor = Ab[j][i] / pivot
            L[j, i] = factor
            # 更新当前行所有元素（从第i列到增广部分）
            for k in range(i, n+1):
                Ab[j][k] -= factor * Ab[i][k]
            steps.append(f"r{j+1} = r{j+1} - ({factor})r{i+1}")
        
        if print_steps:
            print_step_matrix(steps[-1] if steps else "初始矩阵", Ab)
    
    # 分解结果
    U = [[Ab[i][j] for j in range(n)] for i in range(n)]
    b_sym = [row[-1] for row in Ab]
    
    # 回代求解
    x = [Fraction(0) for _ in range(n)]
    for i in reversed(range(n)):
        sum_terms = sum(U[i][j] * x[j] for j in range(i+1, n))
        x[i] = (b_sym[i] - sum_terms) / U[i][i]
    
    return x, P, L, U

def print_step_matrix(step, matrix):
    print(f"\n # {step}")
    print("当前增广矩阵：")
    for row in matrix:
        formatted_row = []
        for num in row:
            if isinstance(num, Fraction):
                if num.denominator == 1:
                    formatted_row.append(f"{num.numerator:4d} ")
                else:
                    formatted_row.append(f"{num.numerator:4d}/{num.denominator:4d}")
            else:
                formatted_row.append(f"{num:8.4f}")
        print("  [" + "  ".join(formatted_row) + "]")
    print("-" * 60)

# 示例用法
if __name__ == "__main__":
    A = [
        [3, -0.1, 1],
        [5, 4, 10],
        [1, 2, 3]
    ]
    b = [2, 0, 1]
    
    x, P, L, U = 列主元消去法符号版(A, b, print_steps=True)
    
    print("\n解为：")
    for i, val in enumerate(x):
        print(f"x{i+1} = {val}")
    
    print("\n分解结果：")
    print("置换矩阵 P:")
    print(P)
    
    print("\n下三角矩阵 L:")
    for row in L:
        print("  ".join(str(Fraction(x).limit_denominator()) for x in row))
    
    print("\n上三角矩阵 U:")
    for row in U:
        print("  ".join(str(Fraction(x).limit_denominator()) for x in row))