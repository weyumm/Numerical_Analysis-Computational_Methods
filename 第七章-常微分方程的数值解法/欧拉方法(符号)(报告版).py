"""
通用欧拉方法（符号运算版）

特点
------
1. Forward Euler y_{n+1} = y_n + h·f(x_n, y_n)
2. 纯 SymPy 符号递推，可选数值替换
3. 一键生成详尽中文报告（纯文本，内含 LaTeX 公式“参考文本”）
4. 基本输入校验与错误处理
5. 便于二次开发：核心求解 + 报告生成 → 松耦合
"""
import sys
from textwrap import dedent
import sympy as sp

class EulerSolver:
    """Forward Euler ODE solver (symbolic)"""

    def __init__(self, f_expr, x0, y0, h, steps, params=None):
        """
        Parameters
        ----------
        f_expr : SymPy Expr      右端函数 f(x, y)
        x0, y0 : number/symbol   初始条件
        h      : float           步长 > 0
        steps  : int             步数 ≥ 0
        params : dict            额外符号参数替换，如 {a: 1, b: 2}
        """
        if h <= 0:
            raise ValueError("步长 h 必须为正数")
        if steps < 0:
            raise ValueError("步数 steps 不能为负")

        self.x, self.y = sp.symbols("x y")
        self.f_expr   = f_expr
        self.x0, self.y0 = sp.nsimplify(x0), sp.nsimplify(y0)
        self.h, self.N  = sp.nsimplify(h), int(steps)
        self.param_subs = params or {}

        # 计算并缓存节点（符号表达式）
        self._xs, self._ys = self._solve_symbolically()

    # ------------------------------------------------------------------ #
    # 核心递推
    # ------------------------------------------------------------------ #
    def _solve_symbolically(self):
        xs, ys = [self.x0], [self.y0]
        for k in range(self.N):
            xi, yi = xs[-1], ys[-1]
            ynext  = sp.simplify(
                yi + self.h * self.f_expr.subs({self.x: xi, self.y: yi})
            )
            xs.append(sp.nsimplify(xi + self.h))
            ys.append(ynext)
        return xs, ys

    # ------------------------------------------------------------------ #
    # 公共接口
    # ------------------------------------------------------------------ #
    def nodes(self):
        """返回 (x_i, y_i_sym) 列表"""
        return list(zip(self._xs, self._ys))

    def numerical(self, subs=None, digits=6):
        """数值化节点值；subs 覆盖 / 追加符号替换"""
        s = {**self.param_subs, **(subs or {})}
        return [(float(sp.N(x.subs(s), digits)),
                 float(sp.N(y.subs(s), digits)))
                for x, y in self.nodes()]

    # ------------------------------------------------------------------ #
    # 报告生成
    # ------------------------------------------------------------------ #
    def report(self, exact=None, digits=6, subs=None):
        """
        返回纯文本报告字符串

        exact : SymPy Expr               精确解 y(x)（若已知）
        subs  : dict                     数值替换，用于打印数值列
        """
        subs_all = {**self.param_subs, **(subs or {})}
        col = 18  # 列宽

        # 头部
        lines = [
            "=" * 22 + " 【欧拉方法】计算报告 " + "=" * 22,
            "递推公式：  y_{n+1} = y_n + h f(x_n, y_n)",
            "",
            f"输入参数：  x0={self.x0},  y0={self.y0},  h={self.h},  N={self.N}",
        ]
        if self.param_subs:
            lines.append("其它参数： " + ", ".join(f"{k}={v}" for k, v in self.param_subs.items()))
        lines.append("")

        # 表头
        header = f"{'n':>3} | {'x_n':^{col}} | {'y_n(符号)':^{col}}"
        if subs_all:
            header += f" | {'y_n(数值)':^{col}}"
        lines += [header, "-" * len(header)]

        # 表体
        for i, (xi, yi) in enumerate(self.nodes()):
            y_sym_str = str(yi) if len(str(yi)) <= col else str(yi)[:col-3] + "..."
            row = f"{i:>3} | {str(xi):^{col}} | {y_sym_str:^{col}}"
            if subs_all:
                y_val = sp.N(yi.subs(subs_all), digits)      # 仍是 SymPy 对象
                y_val_str = f"{float(y_val):.{digits}g}"      # 转成字符串并控制有效数字
                row += f" | {y_val_str:^{col}}"
            lines.append(row)

        # 可选误差分析
        if exact is not None and subs_all:
            lines += ["", "误差分析："]
            for xi, yi in self.nodes():
                y_exact = sp.N(exact.subs({self.x: xi, **subs_all}), digits)
                y_approx = sp.N(yi.subs(subs_all), digits)
                err = y_exact - y_approx
                lines.append(f"  x = {float(sp.N(xi, 3)):<5} → 误差 = {err}")

        lines.append("=" * 70)
        return "\n".join(lines)

# ---------------------------------------------------------------------- #
# 直接运行：给出 3 个示例
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    x, y = sp.symbols("x y")
    a, b = sp.symbols("a b")

    # 示例 1：y' = a·x + b   (纯符号、已知精确解)
    f1 = a*x + b
    solver1 = EulerSolver(f1, x0=0, y0=0, h=0.2, steps=5, params={a: 1, b: 2})
    exact1  = 0.5*a*x**2 + b*x
    print(solver1.report(exact=exact1))

    # 示例 2：y' = y - x^2 + 1   (经典教案，无精确解列)
    f2 = y - x**2 + 1
    solver2 = EulerSolver(f2, x0=0, y0=0.5, h=0.2, steps=5)
    print(solver2.report())

    # 示例 3：y' = sin(x) - y   (含三角函数)
    f3 = sp.sin(x) - y
    solver3 = EulerSolver(f3, x0=0, y0=1, h=0.2, steps=5)
    print(solver3.report(subs={}))

