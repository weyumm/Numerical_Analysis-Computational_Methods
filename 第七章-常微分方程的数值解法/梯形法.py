import numpy as np

def 梯形法(h: float = 0.1, x_end: float = 1.5):
    """
    使用梯形法则计算 [0, x_end] 区间上 y' = -y，y(0) = 1 的数值解。

    返回值
    -------
    x : ndarray
        网格点（大小为 n+1）。
    y : ndarray
        网格点上的数值解。
    """
    # 步数 n 满足 n * h = x_end
    n = int(round(x_end / h))
    x = np.linspace(0.0, x_end, n + 1)

    # 预分配解数组
    y = np.empty(n + 1)
    y[0] = 1.0

    # 预计算常数放大因子 rho = (2 - h) / (2 + h)
    rho = (2.0 - h) / (2.0 + h)

    # 时间步进循环
    for i in range(n):
        y[i + 1] = rho * y[i]        # 梯形格式的单行递推公式

    return x, y


def main():
    h = 0.1
    x, y_num = 梯形法(h)

    # 精确解
    y_exact = np.exp(-x)

    # 打印格式化表格
    header = f"{'i':>2} {'x':>6} {'y_trap':>12} {'y_exact':>12} {'error':>12}"
    print(header)
    print("-" * len(header))
    for i, (xi, yi, ye) in enumerate(zip(x, y_num, y_exact)):
        err = abs(yi - ye)
        print(f"{i:2d} {xi:6.2f} {yi:12.8f} {ye:12.8f} {err:12.2e}")


if __name__ == "__main__":
    main()