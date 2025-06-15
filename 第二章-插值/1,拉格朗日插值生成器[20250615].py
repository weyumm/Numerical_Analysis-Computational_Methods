import numpy as np
from sympy import symbols, expand, lambdify
from sympy import diff, ln, exp, sin, cos, tan, sqrt
import math
import matplotlib.pyplot as plt

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def format_node_table(x_list, y_list):
    """格式化节点表格"""
    headers = ["x", "f(x)"]
    print("【插值节点】")
    print("-" * 60)
    print(" | ".join(headers))
    print("-" * 60)
    for x, y in zip(x_list, y_list):
        print(f"| {x:.2f} | {y:.4f} |")

def generate_basis_functions(selected_x):
    """生成并打印基函数表达式"""
    n = len(selected_x)
    x = symbols('x')
    basis_functions = []
    
    print("\n【基函数表达式】")
    print("-" * 60)
    for k in range(n):
        numerator = 1
        denominator = 1
        terms = []
        
        for i in range(n):
            if i != k:
                terms.append(f"(x - {selected_x[i]:.2f})")
                numerator *= (x - selected_x[i])
                denominator *= (selected_x[k] - selected_x[i])
        
        l_k = numerator / denominator
        basis_functions.append(l_k)
        print(f"l_{k}(x) = {' * '.join(terms)} / {denominator:.4f} = {expand(l_k)}")
    
    return basis_functions

def construct_lagrange_polynomial(selected_x, selected_y):
    """构造拉格朗日插值多项式"""
    x = symbols('x')
    basis_functions = generate_basis_functions(selected_x)
    polynomial = sum(y * l for y, l in zip(selected_y, basis_functions))
    
    print("\n【插值多项式展开过程】")
    print("-" * 60)
    expanded_poly = expand(polynomial)
    print(f"P(x) = {expanded_poly}")
    return expanded_poly

def calculate_interpolation(x_val, selected_x, selected_y, polynomial):
    """计算插值点并输出详细过程"""
    x = symbols('x')
    poly_func = lambdify(x, polynomial, 'numpy')
    final_value = poly_func(x_val)
    
    print("\n【代入计算过程】")
    print("-" * 60)
    print(f"计算 P({x_val:.2f}) 的过程：")
    
    for k in range(len(selected_x)):
        product = 1
        for i in range(len(selected_x)):
            if i != k:
                product *= (x_val - selected_x[i]) / (selected_x[k] - selected_x[i])
        term_value = selected_y[k] * product
        print(f"项 {k+1}: {selected_y[k]:.4f} * {product:.4f} = {term_value:.6f}")
    
    print("-" * 60)
    print(f"最终结果: P({x_val:.2f}) ≈ {final_value:.6f}")
    return final_value

def calculate_truncation_error(x, selected_x, selected_y, n_points, func_expr=None):
    """计算截断误差"""
    x_sym = symbols('x')
    product_term = 1.0
    
    for xi in selected_x:
        product_term *= (x - xi)
    
    if func_expr:
        nth_derivative = diff(func_expr, x_sym, n_points)
        max_deriv = abs(nth_derivative.subs(x_sym, min(selected_x)))
        for xi in selected_x:
            current = abs(nth_derivative.subs(x_sym, xi))
            if current > max_deriv:
                max_deriv = current
                
        error_bound = abs(product_term) / math.factorial(n_points) * max_deriv
        print(f"\n【截断误差估计】")
        print("-" * 60)
        print(f"R_{n_points-1}(x) = f^{n_points}(ξ)/{n_points}! * Π(x-x_i)")
        print(f"误差上界: {error_bound:.6e}")
    
    return product_term

def plot_interpolation(x_list, y_list, selected_x, polynomial, x_unknown):
    """可视化插值结果"""
    x = symbols('x')
    poly_func = lambdify(x, polynomial, 'numpy')
    x_vals = np.linspace(min(x_list) - 0.5, max(x_list) + 0.5, 400)
    y_vals = poly_func(x_vals)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label="插值多项式", color='blue')
    plt.scatter(x_list, y_list, color='red', label="原始数据点")
    plt.scatter(x_unknown, poly_func(x_unknown), color='green', label=f"插值点 x={x_unknown:.2f}")
    
    # 添加坐标标注
    for xi, yi in zip(x_list, y_list):
        plt.annotate(f"({xi:.1f}, {yi:.2f})", (xi, yi), 
                    textcoords="offset points", xytext=(0, 10), 
                    ha='center')
    
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("拉格朗日插值曲线")
    plt.legend()
    plt.grid(True)
    plt.show()

def main(x_list, y_list, x, num_points=3, func_expr=None):
    """主函数"""
    print("【输入数据】")
    print(f"x_list = {x_list}")
    print(f"y_list = {y_list}")
    print(f"插值点 x = {x:.2f}")
    print(f"使用 {num_points} 个节点进行插值")
    print("-" * 60)
    
    # 格式化显示节点表
    format_node_table(x_list, y_list)
    
    # 选择最近的节点
    distances = np.abs(x_list - x)
    sorted_indices = np.argsort(distances)
    selected_indices = sorted_indices[:num_points]
    selected_x = x_list[selected_indices]
    selected_y = y_list[selected_indices]
    
    print(f"\n【使用节点】x = {', '.join([f'{xi:.1f}' for xi in selected_x])}")
    
    # 构造插值多项式
    polynomial = construct_lagrange_polynomial(selected_x, selected_y)
    
    # 计算插值点
    calculate_interpolation(x, selected_x, selected_y, polynomial)
    
    # 计算截断误差
    if func_expr:
        calculate_truncation_error(x, selected_x, selected_y, num_points, func_expr)
    
    # 绘制插值曲线
    plot_interpolation(x_list, y_list, selected_x, polynomial, x)

# 示例调用
if __name__ == "__main__":
    x_list = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    y_list = np.array([1.0000, 1.2214, 1.4918, 1.8221, 2.2255])
    
    # 示例：对 x=0.12 进行 3 点插值
    main(x_list, y_list, x=0.12, num_points=3)
    
    # 示例：对 x=0.3 进行 4 点插值
    # main(x_list, y_list, x=0.3, num_points=4)