import math
import scipy.integrate

def basic_trapezoidal_report(f, a, b, func_str):
    """
    使用基本梯形公式计算积分并生成详细报告

    参数:
    f : 被积函数
    a : 积分下限
    b : 积分上限
    func_str : 函数表达式字符串
    """
    h = b - a
    x_values = [a, b]
    f_values = [f(a), f(b)]

    # 报告标题
    print(f"【基本梯形公式积分报告】\n函数: f(x) = {func_str}\n积分区间: [{a}, {b}]\n")

    # 显示公式推导
    print("一、基本梯形公式：")
    print(f"T = (h/2) * [f(a) + f(b)] 其中 h = b - a = {h}")

    # 计算积分值
    integral = (f(a) + f(b)) * h / 2

    # 显示计算过程
    print("\n二、数值计算：")
    print(f"f({a}) = {f_values[0]:.5f}")
    print(f"f({b}) = {f_values[1]:.5f}")
    print(f"T = ({h}/2) * ({f_values[0]:.5f} + {f_values[1]:.5f})")
    print(f"   = {h/2:.5f} * {sum(f_values):.5f}")
    print(f"   ≈ {integral:.5f}\n")

    # 计算精确值
    true_value, _ = scipy.integrate.quad(f, a, b)
    error = abs(integral - true_value)

    # 结果对比
    print("三、精度分析：")
    print(f"精确值: {true_value:.10f}")
    print(f"近似值: {integral:.10f}")
    print(f"绝对误差: {error:.2e}")
    print("-" * 60 + "\n")

    return integral

# 示例测试
if __name__ == "__main__":
    # 测试1：线性函数
    basic_trapezoidal_report(lambda x: 2*x, 0, 2, "2x")
    
    # 测试2：二次函数
    basic_trapezoidal_report(lambda x: x**2, 0, 1, "x^2")
    
    # 测试3：三角函数
    basic_trapezoidal_report(math.sin, 0, math.pi, "sin(x)")