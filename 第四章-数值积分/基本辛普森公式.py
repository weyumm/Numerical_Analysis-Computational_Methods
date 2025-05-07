import math
import scipy.integrate

def basic_simpson_report(f, a, b, func_str):
    """
    使用基本辛普森公式计算积分并生成详细报告

    参数:
    f : 被积函数
    a : 积分下限
    b : 积分上限
    func_str : 函数表达式字符串
    """
    h = b - a
    mid = (a + b) / 2
    x_values = [a, mid, b]
    f_values = [f(a), f(mid), f(b)]

    # 报告标题
    print(f"【基本辛普森公式积分报告】\n函数: f(x) = {func_str}\n积分区间: [{a}, {b}]\n")

    # 显示公式推导
    print("一、辛普森公式推导：")
    print(f"S = (h/6) * [f(a) + 4f((a+b)/2) + f(b)]")
    print(f"其中 h = b - a = {h:.5f}")
    print(f"中间点 x_m = (a + b)/2 = {mid:.5f}\n")

    # 显示计算过程
    print("二、数值计算：")
    print(f"f({a:.5f}) = {f_values[0]:.5f}")
    print(f"f({mid:.5f}) = {f_values[1]:.5f}")
    print(f"f({b:.5f}) = {f_values[2]:.5f}")

    # 计算积分值
    integral = (h / 6) * (f_values[0] + 4 * f_values[1] + f_values[2])
    
    print(f"\nS = ({h:.5f}/6) * ({f_values[0]:.5f} + 4*{f_values[1]:.5f} + {f_values[2]:.5f})")
    print(f"   = {h/6:.5f} * {f_values[0] + 4*f_values[1] + f_values[2]:.5f}")
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
    # 测试1：二次函数
    basic_simpson_report(lambda x: x**2, 0, 1, "x^2")
    
    # 测试2：四次函数
    basic_simpson_report(lambda x: x**4, 0, 2, "x^4")
    
    # 测试3：三角函数
    basic_simpson_report(math.cos, 0, math.pi/2, "cos(x)")
    
    # 测试4：指数函数
    basic_simpson_report(lambda x: math.exp(x), 0, 1, "e^x")