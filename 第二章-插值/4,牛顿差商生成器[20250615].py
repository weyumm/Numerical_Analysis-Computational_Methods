import numpy as np
from sympy import symbols, expand, lambdify
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def newton_interpolation(x_list, y_list, x):
    """牛顿插值多项式"""
    n = len(x_list)

def generate_divided_diff_table(x_list, y_list):
    """生成牛顿插商表，并记录每阶差商的计算过程"""
    n = len(x_list)
    diff_table = np.zeros((n, n)) * np.nan
    diff_table[:, 0] = y_list
    steps = []

    for j in range(1, n):
        for i in range(n - j):
            numerator = diff_table[i + 1, j - 1] - diff_table[i, j - 1]
            denominator = x_list[i + j] - x_list[i]
            diff_table[i, j] = numerator / denominator

            # 构建差商标签
            indices = [str(i) for i in range(i, i + j + 1)]
            tag = "f[" + ",".join(indices) + "]"

            # 记录计算过程（使用 f[X0,X1] 形式）
            step = f"{tag} = ({diff_table[i+1, j-1]:.4f} - {diff_table[i, j-1]:.4f}) / ({x_list[i+j]:.1f} - {x_list[i]:.1f}) = {diff_table[i, j]:.4f}"
            steps.append(step)

    return diff_table, steps

def print_divided_diff_table(x_list, diff_table, steps):
    """打印差商表和计算过程"""
    n = len(x_list)
    headers = ["x", "f(x)"]
    for i in range(1, n):
        headers.append(f"{i}阶差商")

    print("【差商表】")
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

    print("\n【差商计算过程】")
    print("-" * 60)
    for step in steps:
        print(step)

def select_nodes(x, x_list, num_points):
    """选择距离x最近的num_points个节点，并保持原始顺序"""
    distances = np.abs(x_list - x)
    indices = np.argsort(distances)[:num_points]
    sorted_indices = np.sort(indices)  # 保持原始顺序
    return x_list[sorted_indices], sorted_indices

def format_newton_term(coeffs, x_values, degree):
    """格式化单个牛顿插值项"""
    term = f"{coeffs[degree]:.4f}"
    for i in range(degree):
        term += f"(x - {x_values[i]:.1f})"
    return term

def format_polynomial(coeffs, x_values):
    """格式化整个插值多项式"""
    terms = []
    for i in range(len(coeffs)):
        term = format_newton_term(coeffs, x_values, i)
        terms.append(term)
    return " + ".join(terms)

def expand_and_print_polynomial(coeffs, x_values):
    """将牛顿插值多项式展开为标准多项式形式，并逐步输出过程"""
    x = symbols('x')
    poly = 0

    print("\n【插值多项式展开过程】")
    print("-" * 60)

    for i in range(len(coeffs)):
        base = 1
        for j in range(i):
            base *= (x - x_values[j])
        term = coeffs[i] * base
        poly += term

        # 展开当前项
        expanded_term = expand(term)
        print(f"第{i+1}项: {term} = {expanded_term}")

    # 合并同类项
    final_poly = expand(poly)
    print("-" * 60)
    print(f"合并同类项后: N(x) = {final_poly}")
    return final_poly

def print_newton_interpolation(x, x_list, y_list, diff_table, selected_x, selected_indices, coeffs):
    """打印插值过程"""
    print(f"\n【使用节点 x = {', '.join([f'{xi:.1f}' for xi in selected_x])} 构造插值多项式】")
    print("N(x) = ", end="")

    # 打印公式结构
    formula_parts = []
    for i in range(len(selected_x)):
        part = f"{coeffs[i]:.4f}"
        for j in range(i):
            part += f"(x - {selected_x[j]:.1f})"
        formula_parts.append(part)
    print(" + ".join(formula_parts))

    print("\n【代入计算过程】")
    print("-" * 60)
    value = coeffs[0]
    print(f"第1项: {coeffs[0]:.4f}")

    for i in range(1, len(coeffs)):
        product = np.prod([x - selected_x[j] for j in range(i)])
        term_value = coeffs[i] * product
        value += term_value
        print(f"第{i+1}项: {coeffs[i]:.4f} * {product:.4f} = {term_value:.6f}")

    print("-" * 60)
    print(f"最终结果 N({x:.2f}) ≈ {value:.6f}")

def plot_interpolation(x_list, y_list, selected_x, final_poly, x_unknown):
    """绘制插值曲线与原始点"""
    x_sym = symbols('x')
    poly_func = lambdify(x_sym, final_poly, 'numpy')

    x_vals = np.linspace(min(x_list) - 0.2, max(x_list) + 0.2, 400)
    y_vals = poly_func(x_vals)

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label="插值多项式", color='blue')
    plt.scatter(x_list, y_list, color='red', label="原始数据点")
    plt.scatter(x_unknown, poly_func(x_unknown), color='green', label=f"插值点 x={x_unknown:.2f}")

    # 添加坐标标注
    for xi, yi in zip(x_list, y_list):
        plt.annotate(f"({xi:.1f}, {yi:.2f})", (xi, yi), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("牛顿插值曲线与原始数据点")
    plt.legend()
    plt.grid(True)
    plt.show()

def main(x_list, y_list, x, num_points=3):
    """主函数：生成差商表、插值过程、插值多项式"""
    print("【输入数据】")
    print(f"x_list = {x_list}")
    print(f"y_list = {y_list}")
    print(f"插值点 x = {x:.2f}")
    print(f"使用 {num_points} 个节点进行插值")
    print("-" * 60)

    # 生成差商表
    diff_table, steps = generate_divided_diff_table(x_list, y_list)
    print_divided_diff_table(x_list, diff_table, steps)

    # 选择最近的节点
    selected_x, selected_indices = select_nodes(x, x_list, num_points)
    coeffs = [diff_table[selected_indices[0], k] for k in range(num_points)]

    # 展开为标准多项式
    final_poly = expand_and_print_polynomial(coeffs, selected_x)
    print("\n【最终插值多项式】")
    print(f"N(x) = {final_poly}")

    # 打印插值过程
    print_newton_interpolation(x, x_list, y_list, diff_table, selected_x, selected_indices, coeffs)

    # 绘制插值曲线
    plot_interpolation(x_list, y_list, selected_x, final_poly, x)

# 示例调用
if __name__ == "__main__":
    x_list = np.array([100,121,144])
    y_list = np.array([10,11,12])

    # 示例：对 x=1.75 进行 3 点插值,二次插值多项式
    #main(x_list, y_list, x=1.75, num_points=3)
    # 示例：对 x=1.75 进行 4 点插值,三次插值多项式
    main(x_list, y_list, x=115, num_points=3)