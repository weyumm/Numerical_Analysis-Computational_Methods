import numpy as np
import math
from sympy import symbols, diff, ln, exp, sin, cos, tan, sqrt

def generate_lagrange_polynomial(x_list, y_list):
    """生成拉格朗日插值多项式(Markdown格式)"""
    n = len(x_list)
    poly_terms = []
    
    for k in range(n):
        numerator_terms = []
        denominator = 1.0
        
        for i in range(n):
            if i != k:
                numerator_terms.append(f'(x - {x_list[i]:.2f})')
                denominator *= (x_list[k] - x_list[i])
        
        # 使用Markdown格式的分数表示
        numerator_part = ' * '.join(numerator_terms)
        term = f"[{numerator_part}] / {denominator:.4f}"
        poly_terms.append(f"{y_list[k]:.4f} * ({term})")
    
    return " + ".join(poly_terms)

def format_node_table(x_list, y_list):
    """将节点数据格式化为Markdown表格"""
    table = "| xᵢ | f(xᵢ) |\n"
    table += "|----|-------|\n"
    
    for x, y in zip(x_list, y_list):
        table += f"| {x:.2f} | {y:.4f} |\n"
    
    return table

def calculate_truncation_error(x, selected_x, n_points, func_expr=None):
    """计算拉格朗日插值的截断误差"""
    # 计算乘积项 (x-x0)(x-x1)...(x-xn)
    product = 1.0
    for xi in selected_x:
        product *= (x - xi)
    
    if func_expr:
        # 使用sympy计算高阶导数
        x_sym = symbols('x')
        nth_derivative = diff(func_expr, x_sym, n_points)
        error_term = f"{nth_derivative}/math.factorial({n_points}) * {product:.6f}"
        error_value = nth_derivative.subs(x_sym, x)/math.factorial(n_points) * product
        return f"截断误差R_{n_points}(x) = {error_term} ≈ {error_value:.6e}"
    else:
        return f"截断误差项为：f^({n_points})(ξ)/{n_points}! * {product:.6f}"

def calculate_a_posteriori_error(x, x_list, y_list, n_points):
    """计算插值误差的事后估计"""
    if n_points < 2:
        return "无法计算事后误差估计(需要至少2个插值点)"
    
    # 直接计算n_points点插值结果
    distances = np.abs(x_list - x)
    sorted_indices = np.argsort(distances)
    selected_indices = sorted_indices[:n_points]
    selected_x = x_list[selected_indices]
    selected_y = y_list[selected_indices]
    
    # 计算n点插值结果
    val_n = 0.0
    for k in range(n_points):
        product = 1.0
        for i in range(n_points):
            if i != k:
                product *= (x - selected_x[i]) / (selected_x[k] - selected_x[i])
        val_n += selected_y[k] * product
    
    # 计算n-1点插值结果
    selected_indices_minus_1 = sorted_indices[:n_points-1]
    selected_x_minus_1 = x_list[selected_indices_minus_1]
    selected_y_minus_1 = y_list[selected_indices_minus_1]
    
    val_n_minus_1 = 0.0
    for k in range(n_points-1):
        product = 1.0
        for i in range(n_points-1):
            if i != k:
                product *= (x - selected_x_minus_1[i]) / (selected_x_minus_1[k] - selected_x_minus_1[i])
        val_n_minus_1 += selected_y_minus_1[k] * product
    
    error_estimate = abs(val_n - val_n_minus_1)
    return f"事后误差估计: |P_{n_points-1}({x:.2f}) - P_{n_points-2}({x:.2f})| = {error_estimate:.6e}"

def generate_markdown_solution(x, x_list, y_list, n_points, func_expr=None):
    """生成Markdown格式解答过程"""
    # 修改节点选择逻辑：根据x的距离选择最近的n_points个点
    distances = np.abs(x_list - x)
    sorted_indices = np.argsort(distances)
    selected_indices = sorted_indices[:n_points]
    selected_x = x_list[selected_indices]
    selected_y = y_list[selected_indices]
    
    # 生成基函数展开式
    basis_functions = []
    for k in range(n_points):
        numerator = []
        denominator = []
        
        for i in range(n_points):
            if i != k:
                numerator.append(f'(x - {selected_x[i]:.2f})')
                denominator.append(f'({selected_x[k]:.2f} - {selected_x[i]:.2f})')
        
        basis = f"l_{k}(x) = [{' * '.join(numerator)}] / [{' * '.join(denominator)}]"
        basis_functions.append(basis)
    
    # 生成完整多项式
    full_poly = " + ".join([f"{selected_y[k]:.4f} * l_{k}(x)" for k in range(n_points)])
    
    # 生成数值计算过程
    numeric_calculation = []
    final_value = 0.0
    for k in range(n_points):
        product = 1.0
        for i in range(n_points):
            if i != k:
                product *= (x - selected_x[i]) / (selected_x[k] - selected_x[i])
        numeric_calculation.append(f"{selected_y[k]:.4f} * {product:.4f}")
        final_value += selected_y[k] * product

    # 计算截断误差
    truncation_error = calculate_truncation_error(x, selected_x, n_points, func_expr)
    
    # 计算事后误差估计
    if n_points > 1:
        posteriori_error = calculate_a_posteriori_error(x, x_list, y_list, n_points)
    else:
        posteriori_error = "无法计算事后误差估计(需要至少2个插值点)"
    
    markdown = f"""## {n_points}点拉格朗日插值

### 节点选择：
{format_node_table(selected_x, selected_y)}

### 基函数展开：
{"<br>".join(basis_functions)}

### 插值多项式：
P_{n_points-1}(x) = {full_poly}

### 代入x = {x:.2f}计算：
P_{n_points-1}({x:.2f}) = {' + '.join(numeric_calculation)}  
= {final_value:.6f}

### 截断误差分析：
{truncation_error}

### 事后误差估计：
{posteriori_error}
"""
    return markdown

##################### 主程序示例 #########################
x_list = np.array([10, 11, 12, 13])
y_list = np.array([2.3026, 2.3979, 2.4849, 2.5649])

# 输入原函数表达式
func_expr = ln(symbols('x'))  # 例如ln(x)
# 根据不同函数选择以下一种表达式
#func_expr = exp(symbols('x'))    # e^x
# func_expr = 1/symbols('x')     # 1/x
# func_expr = sin(symbols('x'))   # sin(x)
# func_expr = cos(symbols('x'))   # cos(x)
# func_expr = tan(symbols('x'))   # tan(x)
# func_expr = sqrt(symbols('x'))  # √x
# func_expr = symbols('x')**2     # x²
# func_expr = symbols('x')**3 + 2*symbols('x')**2 - 5  # x³ + 2x² - 5

# 生成二点插值解答
markdown_2point = generate_markdown_solution(11.75, x_list, y_list, 2, func_expr)
print("两点插值解答Markdown代码：\n", markdown_2point)

# 生成三点插值解答
markdown_3point = generate_markdown_solution(11.75, x_list, y_list, 3, func_expr)
print("三点插值解答Markdown代码：\n", markdown_3point)

# 生成四点插值解答
markdown_4point = generate_markdown_solution(11.75, x_list, y_list, 4, func_expr)
print("\n四点插值解答Markdown代码：\n", markdown_4point)