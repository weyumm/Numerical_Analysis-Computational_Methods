import numpy as np

# 输入数据
x = np.array([2,4,6,8], dtype=float)
y = np.array([2,11,28,40], dtype=float)
weights = np.array([14,27,12,1], dtype=float)  # 权重

print("输入数据：")
for xi, yi, wi in zip(x, y, weights):
    print(f"x = {xi}, y = {yi:.2f}, weight = {wi}")

# 构造设计矩阵 X（包含常数项）
X = np.column_stack((x, np.ones_like(x)))

# 构造权重矩阵 W
W = np.diag(weights)

print("\n设计矩阵 X：")
print(X)

print("\n权重矩阵 W：")
print(W)

# 计算加权正规方程系数矩阵和右侧向量
XTWX = X.T @ W @ X
XTWy = X.T @ W @ y

print("\nXTWX（加权正规方程左侧）：")
print(XTWX)

print("\nXTWy（加权正规方程右侧）：")
print(XTWy)

# 解正规方程
coeffs = np.linalg.solve(XTWX, XTWy)

a, b = coeffs
print("\n解得参数：")
print(f"斜率 a = {a:.4f}")
print(f"截距 b = {b:.4f}")

# 计算拟合值
y_fit = a * x + b

print("\n拟合结果：")
for xi, yi, yf in zip(x, y, y_fit):
    print(f"x = {xi}, 实际 y = {yi:.2f}, 拟合 y = {yf:.4f}")

# 可视化（可选）
import matplotlib.pyplot as plt

plt.scatter(x, y, color='blue', label='原始数据')
plt.plot(x, y_fit, color='red', label=f'拟合直线: y = {a:.2f}x + {b:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('加权最小二乘法拟合')
plt.grid(True)
plt.show()