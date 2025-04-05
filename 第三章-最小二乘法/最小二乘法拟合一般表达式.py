import numpy as np

def least_squares_fit(x, y, basis_funcs):
    """
    最小二乘法通用拟合
    
    参数:
    x (array): 自变量数据
    y (array): 因变量数据
    basis_funcs (list): 基函数列表，如 [lambda x: 1, lambda x: x, lambda x: x**2]
    
    返回:
    coeffs (array): 系数数组
    y_pred (array): 预测值
    mse (float): 均方误差
    max_deviation (float): 最大偏差
    """
    # 构造设计矩阵
    X = np.column_stack([func(x) for func in basis_funcs])
    
    # 最小二乘求解
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    
    # 计算预测值和误差
    y_pred = X @ coeffs
    errors = y - y_pred
    
    # 计算指标
    mse = np.mean(errors**2)
    max_deviation = np.max(np.abs(errors))
    
    return coeffs, y_pred, mse, max_deviation

# =====================================
# 示例1：二次多项式拟合 y = a + bx^2
# =====================================
x = np.array([19, 25, 31, 38, 44])
y = np.array([19.0, 32.3, 49.0, 73.3, 97.8])

# 定义基函数（常数项 + 平方项）
basis_funcs = [lambda x: np.ones_like(x), lambda x: x**2]

# 执行拟合
coeffs, y_pred, mse, max_dev = least_squares_fit(x, y, basis_funcs)

# 输出结果
print("=== 二次多项式拟合 ===")
print(f"系数: a = {coeffs[0]:.4f}, b = {coeffs[1]:.5f}")
print(f"拟合方程: y = {coeffs[0]:.4f} + {coeffs[1]:.5f}x²")
print(f"均方误差: {mse:.6f}")
print(f"最大偏差: {max_dev:.6f}")

# =====================================
# 示例2：三角函数拟合 y = a + b*sin(x) + c*cos(x)
# =====================================
x_sin = np.linspace(0, 2*np.pi, 10)
y_sin = 3 + 2*np.sin(x_sin) + 1.5*np.cos(x_sin) + 0.1*np.random.randn(10)

basis_trig = [
    lambda x: np.ones_like(x),
    lambda x: np.sin(x),
    lambda x: np.cos(x)
]

coeffs_trig, y_pred_trig, mse_trig, max_dev_trig = least_squares_fit(x_sin, y_sin, basis_trig)

print("\n=== 三角函数拟合 ===")
print(f"系数: a = {coeffs_trig[0]:.4f}, b = {coeffs_trig[1]:.4f}, c = {coeffs_trig[2]:.4f}")
print(f"拟合方程: y = {coeffs_trig[0]:.4f} + {coeffs_trig[1]:.4f}sin(x) + {coeffs_trig[2]:.4f}cos(x)")
print(f"均方误差: {mse_trig:.6f}")