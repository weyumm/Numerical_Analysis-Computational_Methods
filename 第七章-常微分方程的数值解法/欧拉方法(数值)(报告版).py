import numpy as np

class EulerSolver:
    """
    通用欧拉方法求解常微分方程初值问题 y' = f(x, y), y(x0) = y0

    Attributes
    ----------
    f : callable
        微分方程右端函数 f(x, y)
    x0 : float
        初始点 x0
    y0 : float
        初始值 y0
    h : float
        步长
    x_end : float
        终点
    n : int
        步数
    xs, ys, fs : list[float]
        离散节点、近似解及函数值
    """
    def __init__(self, f, x0, y0, h, x_end):
        if h <= 0:
            raise ValueError("步长 h 必须为正数")
        if x_end <= x0:
            raise ValueError("终点 x_end 必须大于 x0")
        self.f = f
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.h = float(h)
        self.x_end = float(x_end)
        # 计算实际步数，确保覆盖到区间终点
        n_float = (self.x_end - self.x0) / self.h
        self.n = int(np.ceil(n_float + 1e-12))  # 避免浮点误差

    # ---------------- 核心求解 ---------------- #
    def solve(self):
        xs = [self.x0]
        ys = [self.y0]
        fs = []
        for _ in range(self.n):
            x_n, y_n = xs[-1], ys[-1]
            f_n = self.f(x_n, y_n)
            fs.append(f_n)
            y_next = y_n + self.h * f_n
            x_next = x_n + self.h
            xs.append(x_next)
            ys.append(y_next)
        # 记录最后一点的 f 值
        fs.append(self.f(xs[-1], ys[-1]))
        self.xs, self.ys, self.fs = xs, ys, fs
        return xs, ys

    # ---------------- 报告生成 ---------------- #
    def generate_report(self, digits: int = 6) -> str:
        if not hasattr(self, "xs"):
            raise RuntimeError("请先调用 solve() 再生成报告")
        ln, add = [], lambda s: ln.append(str(s))
        add("=" * 22 + " 【欧拉方法】计算报告 " + "=" * 22)
        add(f"  步长 h = {self.h}")
        add("  递推公式:  y_{n+1} = y_n + h * f(x_n, y_n)")
        add("")
        header = f"{'n':>4} | {'x_n':>12} | {'y_n':>12} | {'f(x_n,y_n)':>14}"
        add(header)
        add("-" * len(header))
        for i in range(len(self.xs) - 1):
            add(f"{i:4d} | {self.xs[i]:12.{digits}f} | {self.ys[i]:12.{digits}f} | {self.fs[i]:14.{digits}f}")
        # 最后一个节点
        n = len(self.xs) - 1
        add(f"{n:4d} | {self.xs[-1]:12.{digits}f} | {self.ys[-1]:12.{digits}f} | {self.fs[-1]:14.{digits}f}")
        add("")
        add(f"  最终结果:  y({self.xs[-1]}) ≈ {self.ys[-1]:.{digits}f}")
        return "\n".join(ln)


# ======================= 示例 1 ======================= #
print("\n\n示例 1：y' = a x + b, a = 1.5, b = 0.5, y(0)=0, h=0.2, 区间 [0,1]")
a, b = 1.5, 0.5
f1 = lambda x, y: a * x + b
solver1 = EulerSolver(f1, x0=0.0, y0=0.0, h=0.2, x_end=1.0)
solver1.solve()
print(solver1.generate_report(digits=6))

# ======================= 示例 2 ======================= #
print("\n\n示例 2：y' = y - x^2 + 1, y(0)=0.5, h=0.1, 区间 [0,1]")
f2 = lambda x, y: y - x**2 + 1
solver2 = EulerSolver(f2, x0=0.0, y0=0.5, h=0.1, x_end=1.0)
solver2.solve()
print(solver2.generate_report(digits=6))

# ======================= 示例 3 ======================= #
print("\n\n示例 3：Logistic 方程 y' = r y (1 - y/K), r=0.5, K=10, y(0)=2, h=0.2, 区间 [0,2]")
r, K = 0.5, 10
f3 = lambda x, y: r * y * (1 - y / K)
solver3 = EulerSolver(f3, x0=0.0, y0=2.0, h=0.2, x_end=2.0)
solver3.solve()
print(solver3.generate_report(digits=6))
