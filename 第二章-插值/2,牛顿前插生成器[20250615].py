import numpy as np
from sympy import symbols, expand, lambdify
import matplotlib.pyplot as plt

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def generate_forward_diff_table(y_list):
    """生成向前差分表"""
    y = np.asarray(y_list, dtype=float)
    n = len(y)
    diff_table = np.zeros((n, n)) * np.nan
    diff_table[:, 0] = y
    
    for j in range(1, n):
        for i in range(n-j):
            diff_table[i, j] = diff_table[i+1, j-1] - diff_table[i, j-1]
    
    return diff_table

def print_diff_table(x_list, diff_table):
    """打印差分表"""
    n = len(x_list)
    headers = ["x", "f(x)"]
    for i in range(1, n):
        headers.append(f"Δ^{i}f")
    
    print("【差分表】")
    print("-" * 60)
    print(" | ".join(headers))
    print("-" * 60)
    
    for i in range(n):
        row = [f"{x_list[i]:.1f}", f"{diff_table[i, 0]:.4f}"]
        for j in range(1, n - i):
            if not np.isnan(diff_table[i, j]):
                row.append(f"{diff_table[i, j]:.4f}")
            else:
                row.append("")
        print(" | ".join(row))

def select_nodes(x, x_list, n_points):
    """选择以最接近x的左侧节点为起点的连续n_points个节点"""
    sorted_indices = np.argsort(x_list)
    x_list_sorted = x_list[sorted_indices]
    
    # 找到插入点
    idx = np.searchsorted(x_list_sorted, x)
    start = max(0, min(idx - 1, len(x_list_sorted) - n_points))
    
    selected_indices = sorted_indices[start : start + n_points]
    selected_x = x_list[selected_indices]
    
    return selected_x, selected_indices

def construct_forward_polynomial(selected_x, diff_row, h):
    """构造并展开前插多项式"""
    x = symbols('x')
    s = (x - selected_x[0]) / h
    poly = 0
    
    print("\n【插值多项式展开过程】")
    print("-" * 60)
    
    for k in range(len(diff_row)):
        # 构造第k项
        factorial_k = np.math.factorial(k)
        term_s = 1
        for i in range(k):
            term_s *= (s - i)
        
        term = (diff_row[k] / factorial_k) * term_s
        expanded_term = expand(term)
        poly += expanded_term
        
        print(f"第{k+1}项: {term} = {expanded_term}")
    
    final_poly = expand(poly)
    print("-" * 60)
    print(f"合并同类项后: P(x) = {final_poly}")
    return final_poly

def calculate_and_print_result(x_val, selected_x, diff_row, h, polynomial):
    """计算插值点并输出过程"""
    x_sym = symbols('x')
    s = (x_val - selected_x[0]) / h
    value = 0
    
    print("\n【代入计算过程】")
    print("-" * 60)
    print(f"s = (x - x₀)/h = ({x_val:.2f} - {selected_x[0]:.1f}) / {h:.2f} = {s:.4f}")
    
    for k in range(len(diff_row)):
        factorial_k = np.math.factorial(k)
        term_s = 1
        for i in range(k):
            term_s *= (s - i)
        
        term_value = (diff_row[k] / factorial_k) * term_s
        value += term_value
        
        print(f"第{k+1}项: Δ^{k}f/{factorial_k} * product = {diff_row[k]:.4f}/{factorial_k} * {term_s:.4f} = {term_value:.6f}")
    
    poly_func = lambdify(x_sym, polynomial, 'numpy')
    final_value = poly_func(x_val)
    
    print("-" * 60)
    print(f"最终结果 P({x_val:.2f}) ≈ {final_value:.6f}")
    return final_value

def plot_interpolation(x_list, y_list, selected_x, polynomial, x_unknown):
    """绘制插值曲线"""
    x_sym = symbols('x')
    poly_func = lambdify(x_sym, polynomial, 'numpy')
    x_vals = np.linspace(min(x_list) - 0.2, max(x_list) + 0.2, 400)
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
    plt.title("向前差分插值曲线")
    plt.legend()
    plt.grid(True)
    plt.show()

def main(x_list, y_list, x, num_points=3):
    """主函数"""
    print("【输入数据】")
    print(f"x_list = {x_list}")
    print(f"y_list = {y_list}")
    print(f"插值点 x = {x:.2f}")
    print(f"使用 {num_points} 个节点进行插值")
    print("-" * 60)
    
    # 生成差分表
    diff_table = generate_forward_diff_table(y_list)
    print_diff_table(x_list, diff_table)
    
    # 选择节点
    selected_x, selected_indices = select_nodes(x, x_list, num_points)
    print(f"\n【使用节点】x₀ = {', '.join([f'{xi:.1f}' for xi in selected_x])}")
    
    # 获取差分值
    diff_row = [diff_table[selected_indices[0], k] for k in range(num_points)]
    h = selected_x[1] - selected_x[0]  # 计算步长
    print(f"步长 h = {h:.2f}")
    
    # 构造多项式
    polynomial = construct_forward_polynomial(selected_x, diff_row, h)
    
    # 计算插值
    calculate_and_print_result(x, selected_x, diff_row, h, polynomial)
    
    # 绘图
    plot_interpolation(x_list, y_list, selected_x, polynomial, x)

# 示例调用
if __name__ == "__main__":
    x_list = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    y_list = np.array([1.0000, 1.2214, 1.4918, 1.8221, 2.2255])
    
    # 示例：对 x=0.12 进行 4 点插值
    main(x_list, y_list, x=0.12, num_points=4)
    
    # 示例：对 x=0.3 进行 3 点插值
    # main(x_list, y_list, x=0.3, num_points=3)