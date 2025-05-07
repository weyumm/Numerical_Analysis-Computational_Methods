import math
# 在文件开头新增
import scipy.integrate


def composite_trapezoidal_report(f, a, b, n, func_str):
    """
    使用复合梯形公式计算函数 f 在区间 [a, b] 上的积分，
    并输出详细计算报告。

    参数:
    f : callable
        被积函数，接受一个浮点数参数并返回浮点数。
    a : float
        积分下限。
    b : float
        积分上限。
    n : int
        子区间数，必须是正整数。子区间数=点数-1。
    func_str : str
        被积函数的字符串表示，用于报告输出。
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n 必须是正整数")

    h = (b - a) / n
    x_values = [a + i * h for i in range(n + 1)]
    f_values = [f(x) for x in x_values]

    # 输出报告标题
    print(f"【复合梯形公式积分报告】\n函数: f(x) = {func_str}\n积分区间: [{a}, {b}], 子区间数: {n}\n")

    # 输出参数信息
    print(f"使用 {n + 1} 个等距节点，即将区间 [{a}, {b}] 分成 {n} 个子区间，步长为")
    print(f"h = ({b} - {a}) / {n} = {h:.5f}\n")

    # 输出节点与函数值
    print("节点与函数值如下：")
    for i in range(n + 1):
        x = x_values[i]
        fx = f_values[i]
        print(f"x_{i} = {x:.4f} : f(x_{i}) ≈ {fx:.5f}")
    print()

    # 输出复合梯形公式
    print("一、复合梯形公式：")
    print(f"T = (h/2) * [f(x_0) + 2*(Σ_{{i=1}}^{n-1} f(x_i)) + f(x_{n})]")

    # 计算中间和
    sum_mid = sum(f_values[1:-1])
    total = (f_values[0] + 2 * sum_mid + f_values[-1]) * h / 2

    # 输出代入过程
    print("\n代入数值计算：")
    print(f"h = {h:.5f}")
    print(f"f(x_0) = {f_values[0]:.5f}")
    print(f"Σ中间项 = {sum_mid:.5f}")
    print(f"f(x_{n}) = {f_values[-1]:.5f}")
    print(f"\nT = ({h:.5f}/2) * ({f_values[0]:.5f} + 2*({sum_mid:.5f}) + {f_values[-1]:.5f})")
    print(f"  = {h/2:.5f} * { (f_values[0] + 2*sum_mid + f_values[-1]):.5f}")
    print(f"  ≈ {total:.5f}\n")
    print("积分近似值：", total)
    print("-" * 60)

    # ------------------ 对比真实值部分 ------------------
    # 计算真实积分值
    true_value, _ = scipy.integrate.quad(f, a, b)

    # 计算误差
    error = abs(total - true_value)
    relative_error = (error / abs(true_value) * 100) if true_value != 0 else float('inf')

    # 输出对比信息
    print("\n【积分对比】")
    print(f"真实积分值: {true_value:.10f}")
    print(f"数值积分值: {total:.10f}")
    print(f"绝对误差: {error:.2e}")
    print(f"相对误差: {relative_error:.6f}%")
    print("-" * 60 + "\n")
    # ---------------boatchanting-------------------


# 示例使用

def integrand1(x):
    """
    被积函数，计算 x / (4 + x^2) 的值。
    """
    return x / (4 + x**2)

# 调用函数生成报告
composite_trapezoidal_report(integrand1, 3, 6, 8, "x / (4 + x^2)")
'''
def integrand2(x):
    return math.sin(x)

composite_trapezoidal_report(integrand2, 0, math.pi, 6, "sin(x)")

def integrand3(x):
    return x**2

composite_trapezoidal_report(integrand3, 0, 1, 7, "x^2")'''

# 扩展测试 boatchanting
'''
def integrand_1(x):
    return x**3

composite_trapezoidal_report(integrand_1, 0, 2, 8, "x^3")def integrand_2(x):
    return math.sin(x)

composite_trapezoidal_report(integrand_2, 0, math.pi, 10, "sin(x)")
def integrand_3(x):
    return math.exp(-x)

composite_trapezoidal_report(integrand_3, 0, 3, 6, "exp(-x)")
def integrand_4(x):
    return 1 / (1 + x**2)

composite_trapezoidal_report(integrand_4, -1, 1, 8, "1/(1+x^2)")
def integrand_5(x):
    if x < 1:
        return x**2
    else:
        return 2 - x

composite_trapezoidal_report(integrand_5, 0, 2, 10, "分段函数")'''
