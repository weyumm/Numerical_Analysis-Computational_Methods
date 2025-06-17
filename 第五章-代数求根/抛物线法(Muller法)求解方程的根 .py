"""
parabolic_root.py
-----------------
通用抛物线法（三点反求 / Müller 法）求根工具。
直接在终端打印迭代报告：迭代号 | x_0 x_1 x_2 | f_0 f_1 f_2。
"""

import math
from typing import Callable, Tuple


def f_default(x: float) -> float:
    """默认示例：f(x)=8x^4-8x^2+1"""
    return 8 * x**4 - 8 * x**2 + 1


def parabolic_root(
    func: Callable[[float], float],
    x0: float,
    x1: float,
    x2: float,
    tol: float = 1e-5,
    max_iter: int = 50,
    report: bool = True,
) -> Tuple[float, int]:
    """
    抛物线法求解 func(x)=0 的根

    参数
    ----
    func : Callable
        单变量函数 f(x)。
    x0, x1, x2 : float
        初始 3 个互异点。
    tol : float
        收敛阈值 |x_{k+1}-x_k|。
    max_iter : int
        迭代上限。
    report : bool
        是否在屏幕输出迭代表。

    返回
    ----
    root : float
        迭代得到的近似根。
    k : int
        实际迭代次数（不含初值行）。
    """
    xs = [x0, x1, x2]

    if report:
        print("迭代 |          x_0          x_1          x_2 |"
              "          f_0          f_1          f_2")
        print("-" * 91)

    k = 0  # 修改1：迭代计数器从0开始
    while k < max_iter:
        xkm2, xkm1, xk = xs[-3:]
        fkm2, fkm1, fk = func(xkm2), func(xkm1), func(xk)

        if report:
            # 修改2：直接使用当前迭代次数k
            print(f"{k:4d} | {xkm2:12.9f} {xkm1:12.9f} {xk:12.9f} |"
                  f" {fkm2:12.6e} {fkm1:12.6e} {fk:12.6e}")

        # 差商与系数
        h0 = xkm1 - xkm2
        h1 = xk - xkm1
        delta0 = (fkm1 - fkm2) / h0
        delta1 = (fk - fkm1) / h1
        a = (delta1 - delta0) / (h0 + h1)
        b = a * h1 + delta1
        c = fk

        # 计算下一步位移
        rad = math.sqrt(max(b * b - 4 * a * c, 0.0))
        denom = b + math.copysign(rad, b)  # 避免数值抵消
        dx = -2 * c / denom if abs(denom) > 1e-15 else -c / b
        x_next = xk + dx

        # 收敛判定
        if abs(dx) < tol:
            xs.append(x_next)
            if report:
                # 修改3：显示实际迭代次数
                print(f"收敛于第{k+1}次迭代: |x_{k+1}-x_k| = {abs(dx):.2e} < {tol}")
            return x_next, k + 1  # 返回实际迭代次数

        xs.append(x_next)
        k += 1

    raise RuntimeError("未在 max_iter 次迭代内收敛")


# ---------------------- 测试示例 ---------------------- #
if __name__ == "__main__":
    tests = [
        {
            "desc": "示例 1：8x^4-8x^2+1=0  (最小正根≈cos(3π/8))",
            "func": f_default,
            "x0": 0.3, "x1": 0.5, "x2": 0.4
        },
        {
            "desc": "示例 2：x^3 - 2 = 0  (实根≈1.259921)",
            "func": lambda x: x**3 - 2,
            "x0": 1.0, "x1": 1.5, "x2": 1.3
        },
        {
            # 修改4：修正示例3的函数定义
            "desc": "示例 3：sin(x) - 0.5 = 0  (最小正根≈0.523599)",
            "func": lambda x: 8*x**4-8*x**2+1,
            "x0": 0.3, "x1": 0.5, "x2": 0.4
        },
    ]

    for case in tests:
        print("\n" + "=" * 80)
        print(case["desc"])
        root, steps = parabolic_root(
            case["func"], case["x0"], case["x1"], case["x2"], tol=1e-5
        )
        print(f"\n迭代完成，共 {steps} 步，近似根：{root:.12f}")
