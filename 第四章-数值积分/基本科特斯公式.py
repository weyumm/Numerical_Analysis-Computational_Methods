import math
import scipy.integrate

def basic_cotes_report(f, a, b, func_str):
    """
    使用基本柯特斯公式计算积分并生成详细报告
    
    参数:
    f : 被积函数
    a : 积分下限
    b : 积分上限
    func_str : 函数表达式字符串
    """
    # 计算步长和节点
    h = (b - a) / 3  # 四阶柯特斯公式需要3个子区间
    x_values = [a + i*h for i in range(4)]
    f_values = [f(xi) for xi in x_values]
    
    # 报告标题
    print(f"【基本柯特斯公式积分报告】\n函数: f(x) = {func_str}\n积分区间: [{a}, {b}]\n")
    
    # 公式推导部分
    print("一、柯特斯公式推导：")
    print("S = (3h/8) * [f(x₀) + 3f(x₁) + 3f(x₂) + f(x₃)]")
    print(f"其中 h = (b - a)/3 = ({b:.5f} - {a:.5f})/3 = {h:.5f}")
    print(f"节点分布: x₀={x_values[0]:.5f}, x₁={x_values[1]:.5f}, x₂={x_values[2]:.5f}, x₃={x_values[3]:.5f}\n")
    
    # 数值计算过程
    print("二、数值计算：")
    for i, (x, fx) in enumerate(zip(x_values, f_values)):
        print(f"f(x_{i}) = f({x:.5f}) = {fx:.5f}")
    
    # 计算积分值
    integral = (3*h/8) * (f_values[0] + 3*f_values[1] + 3*f_values[2] + f_values[3])
    
    print(f"\nS = (3*{h:.5f}/8) * ({f_values[0]:.5f} + 3*{f_values[1]:.5f} + 3*{f_values[2]:.5f} + {f_values[3]:.5f})")
    print(f"   = {3*h/8:.5f} * {f_values[0] + 3*f_values[1] + 3*f_values[2] + f_values[3]:.5f}")
    print(f"   ≈ {integral:.5f}\n")
    
    # 精确值计算
    true_value, _ = scipy.integrate.quad(f, a, b)
    error = abs(integral - true_value)
    
    # 精度分析
    print("三、精度分析：")
    print(f"精确值: {true_value:.10f}")
    print(f"近似值: {integral:.10f}")
    print(f"绝对误差: {error:.2e}")
    print("-" * 60 + "\n")
    
    return integral

# 示例测试
if __name__ == "__main__":
    # 测试1：三次函数
    basic_cotes_report(lambda x: x**3, 0, 1, "x³")
    
    # 测试2：五次函数
    basic_cotes_report(lambda x: x**5, 0, 2, "x⁵")
    
    # 测试3：三角函数
    basic_cotes_report(math.sin, 0, math.pi/2, "sin(x)")
    
    # 测试4：指数函数
    basic_cotes_report(lambda x: math.exp(x), 0, 1, "e^x")