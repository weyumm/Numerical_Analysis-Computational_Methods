from typing import Sequence, Tuple, Union, Literal
from fractions import Fraction
import matplotlib.pyplot as plt# 绘图
import numpy as np # 绘图
from sympy import (
    symbols, Rational, Matrix, zeros, solve, latex, simplify,
    Symbol, Piecewise, pprint, S
)

RealLike = Union[int, float, Fraction, Rational]


class CubicSplineSymbolic:
    """
    通用符号三次样条插值类，支持三种典型边界条件：
      • 'natural'                    —— S''(x0)=S''(xn)=0
      • 'second'                     —— 给定 S''(x0)=v0, S''(xn)=vn
      • 'first'  (又称 clamped)      —— 给定 S'(x0)=v0,  S'(xn)=vn

    Attributes
    ----------
    x : List[Rational]
        数据点的 x 坐标，必须严格递增。
    y : List[Rational]
        数据点的 y 坐标。
    n : int
        区间数。
    h : List[Rational]
        每个区间的长度。
    M : List[Rational]
        二阶导节点值。
    polys : List[Expr]
        分段三次多项式 sympy 表达式。
    _report_lines : List[str]
        报告文本。

    Methods
    -------
    __init__(self, x: Sequence[RealLike], y: Sequence[RealLike]) -> None
        初始化样条插值类。
    fit(self, bc_type: Literal['natural', 'second', 'first'] = 'natural',
        bc_vals: Tuple[RealLike, RealLike] = (0, 0)) -> None
        生成样条系数。
    _build_report(self, bc_type: str, bc_vals: Tuple[RealLike, RealLike]) -> None
        内部函数：生成 _report_lines 列表。
    generate_report(self) -> str
        返回完整报告字符串（已经在 fit() 中构建）。
    __call__(self, value: RealLike) -> Rational
        样条函数的数值评估（返回 sympy Rational）。
    plot(self)
        绘制样条插值曲线。
    """

    def __init__(self,
                 x: Sequence[RealLike],
                 y: Sequence[RealLike]) -> None:
        self.x = [Rational(r) for r in x]
        self.y = [Rational(r) for r in y]
        if len(self.x) != len(self.y):
            raise ValueError("x 与 y 数组长度必须相等")
        if len(self.x) < 2:
            raise ValueError("至少需要两个数据点")
        if sorted(self.x) != self.x:
            raise ValueError("要求 x 严格递增")
        self.n = len(self.x) - 1          # 区间数
        self.h = [self.x[i+1] - self.x[i] for i in range(self.n)]

        # 计算结果占位
        self.M = None                     # 二阶导节点值
        self.polys = []                   # 分段三次多项式 sympy Expr
        self._report_lines = []           # 报告文本（list[str]）

    # ----------------- 核心公开接口 -----------------

    def fit(self,
            bc_type: Literal['natural', 'second', 'first'] = 'natural',
            bc_vals: Tuple[RealLike, RealLike] = (0, 0)) -> None:
        """
        生成样条系数。

        Parameters
        ----------
        bc_type : 边界条件类型
        bc_vals : (v0, vn)
                  • 若 bc_type = 'second'  → v0 = S''(x0), vn = S''(xn)
                  • 若 bc_type = 'first'   → v0 = S' (x0), vn = S' (xn)
                  • 'natural' 将忽略 bc_vals
        """
        if bc_type not in ('natural', 'second', 'first'):
            raise ValueError("bc_type 仅支持 'natural' | 'second' | 'first'")

        # ---------- 1. 生成三弯矩(对称三对角)线性方程组 ----------
        # 不同边界条件决定未知元个数及附加方程
        if bc_type == 'natural':
            unknown_M = symbols(f"M1:{self.n}")      # M1 ... M_{n-1}
            M0, Mn = S.Zero, S.Zero
            eqs = []

        elif bc_type == 'second':
            v0, vn = map(Rational, bc_vals)
            unknown_M = symbols(f"M1:{self.n}")      # 内部未知
            M0, Mn = Rational(v0), Rational(vn)
            eqs = []

        else:  # 'first'   (clamped)
            # M0, M1, ..., Mn 全部未知
            unknown_M = symbols(f"M0:{self.n+1}")
            M0, Mn = unknown_M[0], unknown_M[-1]
            eqs = []

        # --- 内部节点的三弯矩方程 ---
        for i in range(1, self.n):
            hi, hi1 = self.h[i-1], self.h[i]
            yi1_yi = (self.y[i+1] - self.y[i]) / hi1
            yi_yi_1 = (self.y[i] - self.y[i-1]) / hi
            if bc_type == 'first':
                Mi_1, Mi, Mi1 = unknown_M[i-1], unknown_M[i], unknown_M[i+1]
            else:
                Mi_1 = M0 if i == 1 else unknown_M[i-2]
                Mi = unknown_M[i-1]
                Mi1 = Mn if i == self.n-1 else unknown_M[i]
            eqs.append(
                hi*Mi_1 + 2*(hi + hi1)*Mi + hi1*Mi1 -
                6*(yi1_yi - yi_yi_1)
            )

        # --- 边界条件为一阶导时，需另加两条方程 ---
        if bc_type == 'first':
            h0, hn_1 = self.h[0], self.h[-1]
            # S'(x0)
            eqs.append(
                (self.y[1] - self.y[0]) / h0
                - (2*h0*unknown_M[0] + h0*unknown_M[1]) / 6
                - Rational(bc_vals[0])
            )
            # S'(xn)
            eqs.append(
                (self.y[-1] - self.y[-2]) / hn_1
                + (2*hn_1*unknown_M[-1] + hn_1*unknown_M[-2]) / 6
                - Rational(bc_vals[1])
            )

        # ---------- 2. 求解线性系统 ----------
        sol = solve(eqs, unknown_M, rational=True, simplify=False)
        if sol is None:
            raise RuntimeError("线性方程组无解，请检查输入或边界条件")

        # 重组 M 列表（含两端）
        M_full = []
        for i in range(self.n + 1):
            if i == 0:
                M_full.append(M0 if bc_type != 'first' else sol[unknown_M[0]])
            elif i == self.n:
                M_full.append(Mn if bc_type != 'first' else sol[unknown_M[-1]])
            else:
                idx = i-1 if bc_type != 'first' else i
                M_full.append(sol[unknown_M[idx]])
        self.M = M_full  # 长度 n+1

        # ---------- 3. 生成分段多项式 ----------
        t = symbols('x')
        self.polys.clear()
        for i in range(self.n):
            hi = self.h[i]
            xi, xi1 = self.x[i], self.x[i+1]
            Mi, Mi1 = self.M[i], self.M[i+1]

            # 样条通用表达式
            Si = (
                Mi  * (xi1 - t)**3 / (6*hi) +
                Mi1 * (t - xi)**3 / (6*hi) +
                (self.y[i]   - Mi * hi**2 / 6) * (xi1 - t) / hi +
                (self.y[i+1] - Mi1* hi**2 / 6) * (t - xi) / hi
            )
            self.polys.append(simplify(Si))

        # ---------- 4. 生成可读报告 ----------
        self._build_report(bc_type, bc_vals)

    # ----------------- 报告 -----------------

    def _build_report(self,
                      bc_type: str,
                      bc_vals: Tuple[RealLike, RealLike]) -> None:
        """内部函数：生成 _report_lines 列表"""
        self._report_lines = []  # reset
        ln = self._report_lines.append

        # —— 标题
        ln("=" * 26 + " 【三次样条插值方法】计算报告 " + "=" * 26)
        ln("")

        # —— 节点表
        ln("节点信息:")
        header = "i | " + " | ".join(f"{i:^8}" for i in range(len(self.x)))
        xpos  = "x | " + " | ".join(f"{latex(xi):^8}" for xi in self.x)
        ypos  = "y | " + " | ".join(f"{latex(yi):^8}" for yi in self.y)
        ln(header)
        ln("-"*len(header))
        ln(xpos)
        ln(ypos)
        ln("")

        # —— 边界条件说明
        if bc_type == 'natural':
            ln("边界条件: Natural (S''(x_0)=S''(x_n)=0)")
        elif bc_type == 'second':
            ln(f"边界条件: S''(x_0)={latex(Rational(bc_vals[0]))}, "
               f"S''(x_n)={latex(Rational(bc_vals[1]))}")
        else:
            ln(f"边界条件: S'(x_0)={latex(Rational(bc_vals[0]))}, "
               f"S'(x_n)={latex(Rational(bc_vals[1]))}")
        ln("")

        # —— 输出 M
        ln("求得每个节点的二阶导数 M_i = S''(x_i):")
        ln("  " + ", ".join(f"M_{i} = {latex(self.M[i])}"
                            for i in range(len(self.M))))
        ln("")

        # —— 分段多项式
        ln("各区间三次多项式 S_i(x):")
        t = Symbol('x')
        for i, Si in enumerate(self.polys):
            interval = f"[{latex(self.x[i])}, {latex(self.x[i+1])}]"
            ln(f"  区间 {interval}:")
            ln(f"    S_{i}(x) = {latex(Si)}")
        ln("")

        # —— 拼装整体 S(x)
        pw_parts = []
        for i, Si in enumerate(self.polys):
            cond = f"{latex(self.x[i])} <= x < {latex(self.x[i+1])}" \
                   if i < self.n-1 else \
                   f"{latex(self.x[i])} <= x <= {latex(self.x[i+1])}"
            pw_parts.append(f"{{ {latex(Si)} , {cond} }}")
        ln("整体样条函数:")
        ln("S(x) = piecewise{ " + ", ".join(pw_parts) + " }")
        ln("")

        ln("=" * 70)

    # ----------------- 公共方法 -----------------

    def generate_report(self) -> str:
        """返回完整报告字符串（已经在 fit() 中构建）"""
        if not self._report_lines:
            raise RuntimeError("请先调用 fit() 计算样条")
        return "\n".join(self._report_lines)

    def __call__(self, value: RealLike) -> Rational:
        """样条函数的数值评估（返回 sympy Rational）"""
        if self.polys == []:
            raise RuntimeError("请先调用 fit()")
        value = Rational(value)
        # 找所属区间
        if not (self.x[0] <= value <= self.x[-1]):
            raise ValueError("插值点超出定义域")
        for i in range(self.n):
            if (self.x[i] <= value < self.x[i+1]) or (i == self.n-1):
                return self.polys[i].subs(Symbol('x'), value)
        raise RuntimeError("区间判定失败")
    
    def plot(self):
        """绘制样条插值曲线"""
        if self.polys == []:
            raise RuntimeError("请先调用 fit() 计算样条")

        # 生成插值点
        x_vals = np.linspace(float(self.x[0]), float(self.x[-1]), 1000)
        y_vals = [float(self(value)) for value in x_vals]

        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, label="Cubic Spline")
        plt.scatter(self.x, self.y, color='red', label="Data Points")
        plt.title("Cubic Spline Interpolation")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.show()


def demo():
    # ---------- 示例 1：题目（1）指定二阶导 ----------
    xs = [0, 1, 2, 3,4]
    ys = [0, 0, 0, 0,0]
    spline1 = CubicSplineSymbolic(xs, ys)
    spline1.fit(bc_type='second', bc_vals=(1, 0))
    print(spline1.generate_report())
    spline1.plot()

    # ---------- 示例 2：题目（2）指定一阶导 ----------
    spline2 = CubicSplineSymbolic(xs, ys)
    spline2.fit(bc_type='first', bc_vals=(1, 0))
    print(spline2.generate_report())
    spline2.plot()

    # ---------- 示例 3：自然样条 ----------
    xs2 = [0, 1, 2, 3, 4]
    ys2 = [0, 1, 0, -1, 0]
    spline3 = CubicSplineSymbolic(xs2, ys2)
    spline3.fit(bc_type='natural')
    print(spline3.generate_report())


if __name__ == "__main__":
    demo()
