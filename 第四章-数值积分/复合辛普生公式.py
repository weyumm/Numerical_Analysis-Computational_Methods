import math
# 新增导入
from scipy.integrate import quad

def composite_simpson(f, f_str, a, b, n):
    """
    boatchanting

    使用复合辛普森法则计算定积分 ∫ₐᵇ f(x) dx，并输出详细报告

    参数:
    f (function): 被积函数
    
    f_str (str): 函数表达式字符串，用于报告展示

    a (float): 积分下限

    b (float): 积分上限

    n (int): 子区间数量，必须为偶数

    返回:
    float: 积分的近似值

    示例使用：
    >>> def f(x):
    ...     return x**2
    >>> composite_simpson(f, "x^2", 0, 2, 4)
    """

    if n % 2 != 0:
        raise ValueError("n 必须为偶数")

    h = (b - a) / n
    x_values = [a + i * h for i in range(n + 1)]
    f_values = [f(x) for x in x_values]
    total = 0.0

    # 打印标题
    print("=" * 80)
    print("【定积分计算报告】")
    print("-" * 80)
    print(f"函数表达式: f(x) = {f_str}")
    print(f"积分区间: [{a}, {b}]")
    print(f"子区间数: {n}")
    print(f"节点数: {n + 1}")
    print(f"步长 h = (b - a)/n = ({b} - {a}) / {n} = {h:.5f}")

    # 打印节点与函数值
    print("\n节点与函数值如下：")
    print("-" * 50)
    print("i     x_i          f(x_i)")
    print("-" * 50)
    for i in range(n + 1):
        print(f"{i:<3} | {x_values[i]:<10.5f} | {f_values[i]:<10.5f}")

    # 打印辛普森公式展开
    print("\n复合辛普森公式展开：")
    print("-" * 80)
    print(f"S = h/3 * [f(x_0) + 4*f(x_1) + 2*f(x_2) + 4*f(x_3) + ... + f(x_n)]")
    print("其中，系数规则为：")
    print("  - 第一个和最后一个项系数为 1")
    print("  - 奇数位置项系数为 4")
    print("  - 偶数位置项系数为 2")

    # 构建公式字符串
    terms = []
    for i in range(n + 1):
        if i == 0 or i == n:
            terms.append(f"f(x_{i})")
        elif i % 2 == 1:
            terms.append(f"4*f(x_{i})")
        else:
            terms.append(f"2*f(x_{i})")
    formula = "S = h/3 * [" + " + ".join(terms) + "]"
    print(formula)

    # 构建数值代入过程
    print("\n代入数值计算：")
    print("-" * 80)
    print(f"h/3 = {h:.5f}/3 = {h/3:.5f}")
    print("加权和 = ", end="")

    weighted_sum_str = []
    for i in range(n + 1):
        if i == 0 or i == n:
            coeff = 1
        elif i % 2 == 1:
            coeff = 4
        else:
            coeff = 2
        term = f"{coeff} * {f_values[i]:.5f}"
        weighted_sum_str.append(term)

    print(" + ".join(weighted_sum_str))

    # 计算加权和
    weighted_sum = sum(
        (1 if i == 0 or i == n else 4 if i % 2 == 1 else 2) * f_values[i]
        for i in range(n + 1)
    )

    # 计算积分
    integral = (h / 3) * weighted_sum

    # 输出结果
    print(f"加权和 = {weighted_sum:.5f}")
    print(f"积分 ≈ {h/3:.5f} × {weighted_sum:.5f} = {integral:.5f}")

    print("-" * 80)
    print(f"最终积分结果: ≈ {integral:.5f}")
    print("=" * 80)
    print()

    # 对比逻辑
    # 计算 scipy 精确积分值
    scipy_result, _ = quad(f, a, b)
    
    # 计算绝对误差
    abs_error = abs(integral - scipy_result)

    # 输出对比结果
    print("\n【结果对比】")
    print("-" * 50)
    print(f"复合辛普森法结果: {integral:.10f}")
    print(f"Scipy 精确积分值: {scipy_result:.10f}")
    print(f"绝对误差: {abs_error:.2e}")
    print("=" * 80)
    print()

    return integral, scipy_result

# 示例使用
# 示例 1: ∫₃⁶ x/(4+x²) dx, n = 8
f1 = lambda x: x / (4 + x**2)
composite_simpson(f1, "x / (4 + x^2)", 3, 6, 8)

# 示例 2: ∫₀¹ e^(-x²) dx, n = 4
def f2(x):
    return math.exp(-x**2)
composite_simpson(f2, "e^(-x^2)", 0, 1, 5)

'''

'''
# 扩展测试1
f_test_1 = lambda x: x**3 + 2*x**2 - x + 1
composite_simpson(f_test_1, "x**3 + 2*x**2 - x + 1", 0, 2, 10)