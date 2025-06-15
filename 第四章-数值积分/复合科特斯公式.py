import math
from scipy.integrate import quad

def composite_cotes(f, f_str, a, b, n):
    """
    使用复合科特斯法则计算定积分 ∫ₐᵇ f(x) dx，并输出详细报告
    
    参数:
    f (function): 被积函数
    f_str (str): 函数表达式字符串
    a (float): 积分下限
    b (float): 积分上限
    n (int): 子区间数量（每个子区间被分成3等分）
    
    返回:
    float: 积分的近似值
    """
    if n < 1:
        raise ValueError("子区间数 n 必须为正整数")

    h = (b - a) / n  # 每个子区间的长度
    total_points = 3 * n + 1
    x_values = [a + i * h / 3 for i in range(total_points)]  # 生成3n+1个点
    f_values = [f(x) for x in x_values]
    
    # 初始化系数数组
    coefficients = [0] * total_points
    for k in range(n):
        start = 3 * k
        coefficients[start] += 1
        coefficients[start + 1] += 3
        coefficients[start + 2] += 3
        coefficients[start + 3] += 1

    # 打印标题
    print("=" * 80)
    print("【定积分计算报告 - 复合科特斯公式】")
    print("-" * 80)
    print(f"函数表达式: f(x) = {f_str}")
    print(f"积分区间: [{a}, {b}]")
    print(f"子区间数: {n}")
    print(f"每个子区间等分数: 3")
    print(f"总节点数: {total_points}")
    print(f"步长 h = (b - a)/n = ({b} - {a}) / {n} = {h:.5f}")
    print(f"等分步长 h/3 = {h/3:.5f}")

    # 打印节点与函数值
    print("\n节点与函数值如下：")
    print("-" * 60)
    print("i     x_i          f(x_i)       系数")
    print("-" * 60)
    for i in range(total_points):
        print(f"{i:<3} | {x_values[i]:<10.5f} | {f_values[i]:<10.5f} | {coefficients[i]}")

    # 打印科特斯公式展开
    print("\n复合科特斯公式展开：")
    print("-" * 80)
    print(f"S = (3h/8) * [Σ (系数 * f(x_i))]，其中h为每个子区间的长度")
    print("系数规则：")
    print("  - 首末点系数为1")
    print("  - 每个子区间的中间点系数为3")
    print("  - 相邻子区间的交界点系数累加为2")

    # 构建公式字符串
    terms = []
    for i in range(total_points):
        coeff = coefficients[i]
        if coeff > 0:
            terms.append(f"{coeff}*f(x_{i})")
    formula = "S = (3h/8) * [" + " + ".join(terms) + "]"
    print(formula)

    # 构建数值代入过程
    print("\n代入数值计算：")
    print("-" * 80)
    print(f"3h/8 = 3*{h:.5f}/8 = {3*h/8:.5f}")
    print("加权和 = ", end="")

    weighted_sum_str = []
    for i in range(total_points):
        coeff = coefficients[i]
        if coeff > 0:
            term = f"{coeff} * {f_values[i]:.5f}"
            weighted_sum_str.append(term)

    print(" + ".join(weighted_sum_str))

    # 计算加权和
    weighted_sum = sum(coefficients[i] * f_values[i] for i in range(total_points))

    # 计算积分
    integral = (3 * h / 8) * weighted_sum

    # 输出结果
    print(f"加权和 = {weighted_sum:.5f}")
    print(f"积分 ≈ {3*h/8:.5f} × {weighted_sum:.5f} = {integral:.5f}")
    print("-" * 80)
    print(f"最终积分结果: ≈ {integral:.5f}")
    print("=" * 80)
    print()

    # 对比scipy结果
    scipy_result, _ = quad(f, a, b)
    abs_error = abs(integral - scipy_result)

    # 输出对比结果
    print("\n【结果对比】")
    print("-" * 50)
    print(f"复合科特斯法结果: {integral:.10f}")
    print(f"Scipy 精确积分值: {scipy_result:.10f}")
    print(f"绝对误差: {abs_error:.2e}")
    print("=" * 80)
    print()

    return integral, scipy_result

# 示例使用
if __name__ == "__main__":
    # 示例1：∫₀¹ x³ dx，精确解为0.25，科特斯公式应精确计算
    print("示例1: ∫₀¹ x³ dx")
    f1 = lambda x: x**3
    composite_cotes(f1, "x³", 0, 1, n=1)
    
    # 示例2：∫₀¹ e^(-x²) dx, 精确解无解析式
    print("示例2: ∫₀¹ e^(-x²) dx")
    def f2(x):
        return math.exp(-x**2)
    composite_cotes(f2, "e^(-x²)", 0, 1, n=2)
    
    # 示例3：∫₀² (x³ + 2x² - x + 1) dx, 精确解为 10.666666...
    print("示例3: ∫₀² (x³ + 2x² - x + 1) dx")
    f3 = lambda x: x**3 + 2*x**2 - x + 1
    composite_cotes(f3, "x³ + 2x² - x + 1", 0, 2, n=2)