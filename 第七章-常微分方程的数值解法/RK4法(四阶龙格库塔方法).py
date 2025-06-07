import numpy as np

# 微分方程 y' = x + y
def f(x, y):
    return x + y

# 初值与步长
x0, y0 = 0.0, -1.0
h        = 0.1
x_end    = 2.0

# 生成网格点
xs = np.arange(x0, x_end + h, h)       # 包含终点
ys = np.zeros_like(xs)
ys[0] = y0

# RK-4 主循环
for i in range(len(xs) - 1):
    x, y = xs[i], ys[i]
    k1 = f(x, y)
    k2 = f(x + h/2, y + h/2 * k1)
    k3 = f(x + h/2, y + h/2 * k2)
    k4 = f(x + h,   y + h   * k3)
    ys[i + 1] = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)

# 精确解与误差
exact  = -xs - 1
errors = np.abs(ys - exact)

# 输出结果
print(f"{'x':>4} {'RK4':>14} {'exact':>14} {'|error|':>14}")
for x, y_num, y_exact, err in zip(xs, ys, exact, errors):
    print(f"{x:>4.1f} {y_num:>14.10f} {y_exact:>14.10f} {err:>14.3e}")

print("\nmax abs error =", errors.max())
