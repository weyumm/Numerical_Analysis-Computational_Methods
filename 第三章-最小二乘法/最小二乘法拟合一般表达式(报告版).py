import numpy as np

def least_squares_report(x, y, basis_funcs_with_names):
    """
    最小二乘法拟合并生成报告
    
    参数:
    x (array): 自变量数据
    y (array): 因变量数据
    basis_funcs_with_names (list): 基函数及其名称的元组列表，如 [(lambda x: 1, "1"), (lambda x: x**2, "x²")]
    """
    # 提取基函数和名称
    funcs = [f[0] for f in basis_funcs_with_names]
    func_names = [f[1] for f in basis_funcs_with_names]
    n = len(funcs)
    
    # 构造设计矩阵
    X = np.column_stack([func(x) for func in funcs])
    
    # 计算法方程组矩阵和右侧向量
    XtX = X.T @ X
    Xty = X.T @ y
    
    # 解方程组
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    
    # 计算预测值和误差
    y_pred = X @ coeffs
    errors = y - y_pred
    mse = np.mean(errors**2)
    max_dev = np.max(np.abs(errors))
    
    # 生成报告
    # 法方程组符号形式
    print("法方程组：")
    print("[")
    for i in range(n):
        row = []
        for j in range(n):
            row.append(f"sum({func_names[i]}*{func_names[j]})")
        print("  [" + ", ".join(f"{entry:<12}" for entry in row) + "],")
    print("]")
    print("右侧向量：")
    print("[")
    for i in range(n):
        print(f"  sum({func_names[i]}*y)")
    print("]\n")
    
    # 代入数值
    print("代入具体数值：")
    print("[")
    for i in range(n):
        row = XtX[i].astype(int)  # 转换为整数显示
        print(f"  [{row[0]:<6d}, {row[1]:<7d}],")
    print("]")
    print("右侧向量：")
    print("[")
    for val in Xty:
        print(f"  {val:.1f}")
    print("]\n")
    
    # 解方程组结果
    print("解方程组：")
    for i in range(n):
        print(f"{['a', 'b'][i]:>2} ≈ {coeffs[i]:.4f}")
    
    # 拟合多项式
    print("\n拟合多项式：")
    poly_terms = []
    for i in range(n):
        if coeffs[i] < 0:
            poly_terms.append(f"{coeffs[i]:.4f}{func_names[i]}")
        else:
            poly_terms.append(f"+ {coeffs[i]:.4f}{func_names[i]}")
    poly_str = "y = " + " ".join(poly_terms).replace("+ -", "- ")
    print(poly_str)
    
    # 预测值与误差
    print("\n预测值与误差：")
    print(f"{'x':<4} | {'y实际':<5} | {'y预测':<8} | {'误差':<8}")
    for xi, yi, yp in zip(x, y, y_pred):
        print(f"{xi:<4} | {yi:<5.1f} | {yp:<8.4f} | {yi - yp:<8.4f}")
    
    # 误差指标
    print(f"\n均方误差：{mse:.6f}")
    print(f"最大偏差：{max_dev:.6f}")

# 示例数据
x = np.array([19, 25, 31, 38, 44])
y = np.array([19.0, 32.3, 49.0, 73.3, 97.8])

# 定义基函数及其名称
basis_funcs = [
    (lambda x: np.ones_like(x), "1"),
    (lambda x: x**2, "x²")
]

# 生成报告
least_squares_report(x, y, basis_funcs)