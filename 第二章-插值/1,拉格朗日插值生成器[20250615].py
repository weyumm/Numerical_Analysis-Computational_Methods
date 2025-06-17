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

def calculate_truncation_error(x_val, selected_x, selected_y, n_points, func_expr=None):
    """计算截断误差"""
    if func_expr is None:
        print("\n【截断误差估计】")
        print("-" * 60)
        print("无法计算截断误差：未提供原始函数表达式")
        return None
    
    x_sym = symbols('x')
    
    # 计算乘积项: (x - x0)(x - x1)...(x - xn)
    product_term = 1.0
    for xi in selected_x:
        product_term *= (x_val - xi)
    
    # 计算n_points阶导数
    try:
        nth_derivative = diff(func_expr, x_sym, n_points)
    except Exception as e:
        print(f"计算导数时出错: {e}")
        return None
    
    print(f"\n【截断误差估计】")
    print("-" * 60)
    print(f"R_{n_points-1}(x) = f^{n_points}(ξ)/{n_points}! * Π(x-x_i)")
    
    # 在区间[min(selected_x), max(selected_x)]上寻找导数的最大值
    interval_min = min(selected_x)
    interval_max = max(selected_x)
    
    # 在区间内生成100个点评估导数
    test_points = np.linspace(interval_min, interval_max, 100)
    deriv_func = lambdify(x_sym, nth_derivative, 'numpy')
    
    max_deriv = -np.inf
    for pt in test_points:
        try:
            deriv_val = abs(deriv_func(pt))
            if deriv_val > max_deriv:
                max_deriv = deriv_val
        except:
            continue
    
    if max_deriv == -np.inf:
        print("警告：无法在区间内计算导数最大值")
        max_deriv = 1.0
    
    # 计算误差上界
    factorial_term = math.factorial(n_points)
    error_bound = abs(product_term) * max_deriv / factorial_term
    
    print(f"乘积项: |Π(x-x_i)| = {abs(product_term):.6e}")
    print(f"f^{n_points}(x)在区间[{interval_min:.2f}, {interval_max:.2f}]内的最大绝对值: {max_deriv:.6e}")
    print(f"{n_points}! = {factorial_term}")
    print(f"误差上界: |R_{n_points-1}({x_val:.2f})| ≤ {error_bound:.6e}")
    
    return error_bound

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
    interpolated_value = calculate_interpolation(x, selected_x, selected_y, polynomial)
    
    # 计算截断误差
    error_bound = calculate_truncation_error(x, selected_x, selected_y, num_points, func_expr)
    
    # 如果原始函数表达式已知，计算真实误差
    if func_expr and interpolated_value is not None:
        true_func = lambdify(symbols('x'), func_expr, 'numpy')
        try:
            true_value = true_func(x)
            actual_error = abs(true_value - interpolated_value)
            print(f"\n【实际误差】")
            print("-" * 60)
            print(f"真实值: f({x:.2f}) = {true_value:.6f}")
            print(f"插值值: P({x:.2f}) = {interpolated_value:.6f}")
            print(f"实际误差: |真实值 - 插值值| = {actual_error:.6e}")
            
            if error_bound is not None:
                print(f"误差上界与实际误差的比例: {error_bound/actual_error:.2f}倍")
        except Exception as e:
            print(f"计算真实值时出错: {e}")
    
    # 绘制插值曲线
    plot_interpolation(x_list, y_list, selected_x, polynomial, x)

# 示例调用
if __name__ == "__main__":
    x_list = np.array([10, 11, 12, 13])
    y_list = np.array([2.3026, 2.3979, 2.4849, 2.5649])
    
    # 定义原始函数 (ln(x))
    x_sym = symbols('x')
    func_expr = ln(x_sym)
    
    # 示例：对 x=11.75 进行 3 点插值
    main(x_list, y_list, x=11.75, num_points=3, func_expr=func_expr)