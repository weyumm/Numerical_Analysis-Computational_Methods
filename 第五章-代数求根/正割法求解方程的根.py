import math

# ──────────────────────────────────────────────────────────────
# 1. 正割法核心函数
# ──────────────────────────────────────────────────────────────
def secant_method(f, x0, x1, tol=1e-5, max_iter=100):
    """
    使用正割法求 f(x)=0 的近似根
    Parameters
    ----------
    f        : callable, 目标函数
    x0, x1   : float,   两个不同的初始猜测
    tol      : float,   收敛阈值，默认为 1e-5
    max_iter : int,     迭代次数上限
    Returns
    -------
    root     : float,   近似根
    history  : list[tuple], 每步记录 (k, x_k, |Δx|)
    """
    history = [(0, x0, None), (1, x1, abs(x1 - x0))]

    for k in range(2, max_iter + 2):           # 从第 2 次迭代开始
        f0, f1 = f(x0), f(x1)
        if f1 == f0:                            # 避免分母为 0
            raise ZeroDivisionError("f(x1) 与 f(x0) 极度接近，无法继续迭代")

        # 正割公式
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        delta = abs(x2 - x1)
        history.append((k, x2, delta))

        if delta < tol:                         # 收敛
            return x2, history

        # 更新两点
        x0, x1 = x1, x2

    raise RuntimeError(f"超过最大迭代次数 {max_iter} 仍未收敛")


# ──────────────────────────────────────────────────────────────
# 2. 报告生成函数
# ──────────────────────────────────────────────────────────────
def build_report(f, x0, x1, tol=1e-5):
    """
    boatchanting
    功能 :
      1) 用正割法求解 f(x)=0 的实根
      2) 输出整洁可读的迭代报告 (纯文本，无 LaTeX)
      3) 适配任意函数 / 初值 / 收敛精度
    执行正割法并返回格式化的纯文本报告

    Parameters
    ----------
    f        : callable, 目标函数
    x0, x1   : float,   两个不同的初始猜测
    tol      : float,   收敛阈值，默认为 1e-5
    Returns
    -------
    report   : str,     完整的迭代报告文本

    示例
    ----
    >>> report = build_report(lambda x: x**4 - 3*x + 1, 0.3, 0.4)
    >>> print(report)
    """
    root, hist = secant_method(f, x0, x1, tol)

    # 组织报告文本
    lines = []
    lines.append("================================================================")
    lines.append("                        正  割  法  报  告")
    lines.append("================================================================")
    lines.append("公式:  x_{k+1} = x_k - f(x_k)*(x_k - x_{k-1}) / (f(x_k) - f(x_{k-1}))")
    lines.append(f"初始值: x0 = {x0},  x1 = {x1}")
    lines.append(f"判据  : |x_(k+1) - x_k| < {tol}")
    lines.append("\n迭代过程:")
    lines.append("  k    x_k                  |Δx|")
    lines.append("----  -------------------  -----------")

    # 打印迭代详情
    for k, xk, dx in hist[1:]:                  # 跳过 k=0 的 None 行
        lines.append(f"{k:>4d}  {xk:>19.10f}  {dx:>11.2e}")

    # 摘要
    lines.append("\n结果:")
    lines.append(f"  迭代步数 (不含首两步) : {len(hist)-2}")
    lines.append(f"  近似根 x             : {root:.10f}")
    lines.append(f"  f(x)                 : {f(root):.2e}")
    lines.append("================================================================\n")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# 3. 测试示例
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        # (函数, 初值 x0, 初值 x1, 描述)
        (lambda x: x**4 - 3*x + 1, 0.3, 0.4, "示例 1 :  x^4 - 3x + 1 = 0"),
        (lambda x: math.cos(x) - x,     0.0, 1.0, "示例 2 :  cos(x) - x = 0"),
        (lambda x: x**3 - 2*x - 5, 2.0, 3.0, "示例 3 :  x^3 - 2x - 5 = 0"),
    ]

    for f, x0, x1, desc in tests:
        print(desc)
        try:
            report = build_report(f, x0, x1, tol=1e-5)
            print(report)
        except Exception as e:
            print(f"  发生错误: {e}\n")
