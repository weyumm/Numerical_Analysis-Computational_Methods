import numpy as np

def cubic_spline_m(x, y, M0, M3):
    n = len(x)
    h = [x[i+1] - x[i] for i in range(n-1)]
    
    # 构造方程组求解中间节点的二阶导数
    A = np.array([[2, 0.5], [0.5, 2]])
    B = np.array([-0.5*M0, -0.5*M3])
    M1, M2 = np.linalg.solve(A, B)
    M = [M0, M1, M2, M3]
    
    # 定义分段三次样条函数
    def S(x_val):
        for i in range(n-1):
            if x[i] <= x_val <= x[i+1]:
                t = x_val - x[i]
                hi = h[i]
                Mi = M[i]
                Mi1 = M[i+1]
                term1 = (Mi * (hi - t)**3 + Mi1 * t**3) / (6 * hi)
                term2 = (y[i] - Mi * hi**2 / 6) * (1 - t/hi)
                term3 = (y[i+1] - Mi1 * hi**2 / 6) * (t/hi)
                return term1 + term2 + term3
        return 0.0
    return S

def cubic_spline_derivative(x, y, left_deriv, right_deriv):
    n = len(x)
    h = [x[i+1] - x[i] for i in range(n-1)]
    
    # 构造线性方程组
    A = np.zeros((9, 9))
    B = np.zeros(9)
    
    # 方程1-3: S_i(x_{i+1}) = 0
    for i in range(3):
        A[i, 3*i] = h[i]
        A[i, 3*i+1] = h[i]**2
        A[i, 3*i+2] = h[i]**3
        B[i] = 0
    
    # 方程4-5: 导数连续性
    A[3, 0] = 1
    A[3, 1] = 2*h[0]
    A[3, 2] = 3*h[0]**2
    A[3, 3] = -1
    B[3] = 0
    
    A[4, 3] = 1
    A[4, 4] = 2*h[1]
    A[4, 5] = 3*h[1]**2
    A[4, 6] = -1
    B[4] = 0
    
    # 方程6-7: 二阶导数连续性
    A[5, 1] = 1
    A[5, 2] = 3*h[0]
    A[5, 4] = -1
    B[5] = 0
    
    A[6, 4] = 1
    A[6, 5] = 3*h[1]
    A[6, 7] = -1
    B[6] = 0
    
    # 方程8: 左边界条件
    A[7, 0] = 1
    B[7] = left_deriv
    
    # 方程9: 右边界条件
    A[8, 6] = 1
    A[8, 7] = 2*h[2]
    A[8, 8] = 3*h[2]**2
    B[8] = right_deriv
    
    # 解方程
    coeffs = np.linalg.solve(A, B)
    b = coeffs[::3]
    c = coeffs[1::3]
    d = coeffs[2::3]
    
    # 定义分段三次样条函数
    def S(x_val):
        for i in range(n-1):
            if x[i] <= x_val <= x[i+1]:
                t = x_val - x[i]
                return b[i]*t + c[i]*t**2 + d[i]*t**3
        return 0.0
    return S

# 输入数据
x = [0, 1, 2, 3]
y = [0, 0, 0, 0]

# 问题(1) S''(0)=1, S''(3)=0
spline1 = cubic_spline_m(x, y, M0=1, M3=0)

# 问题(2) S'(0)=1, S'(3)=0
spline2 = cubic_spline_derivative(x, y, left_deriv=1, right_deriv=0)

# 测试样条函数
test_points = [0.5, 1.5, 2.5]
print("问题(1)结果：")
for p in test_points:
    print(f"S({p}) = {spline1(p):.6f}")

print("\n问题(2)结果：")
for p in test_points:
    print(f"S({p}) = {spline2(p):.6f}")