import numpy as np

def generate_divided_diff_table(x_list, y_list):
    """生成牛顿插商表"""
    n = len(x_list)
    diff_table = np.zeros((n, n)) * np.nan
    diff_table[:, 0] = y_list
    
    for j in range(1, n):
        for i in range(n-j):
            diff_table[i, j] = (diff_table[i+1, j-1] - diff_table[i, j-1]) / (x_list[i+j] - x_list[i])
            
    return diff_table

def format_divided_diff_table(x_list, diff_table):
    """将插商表格式化为Markdown表格"""
    header = ["x", "e^x"]
    max_order = diff_table.shape[1] - 1
    header += [f"{i+1}阶差商" for i in range(max_order)]
    
    # 生成表头
    markdown = "| " + " | ".join(header) + " |\n"
    markdown += "|" + "|".join(["----"] * (max_order + 2)) + "|\n"
    
    # 生成表格内容
    for i in range(len(x_list)):
        row = [f"{x_list[i]:.1f}"]
        for j in range(diff_table.shape[1]):
            if np.isnan(diff_table[i, j]):
                row.append("")
            else:
                fmt = "{:.4f}" if j == 0 else "{:.4f}"
                row.append(fmt.format(diff_table[i, j]))
        markdown += "| " + " | ".join(row) + " |\n"
        
    return markdown

def select_nodes_divided(x, x_list, n_points):
    """根据x值选择最近的n_points个节点"""
    distances = np.abs(x_list - x)
    sorted_indices = np.argsort(distances)
    selected_indices = sorted_indices[:n_points]
    selected_x = x_list[selected_indices]
    return selected_x, selected_indices

def generate_latex_solution_divided(x, x_list, diff_table, n_points):
    """生成LaTeX解答过程（牛顿插商）"""
    # 修改节点选择逻辑：根据x的距离选择最近的n_points个点
    selected_x, selected_indices = select_nodes_divided(x, x_list, n_points)
    
    # 获取使用的插商项（从第一个选中的节点开始）
    used_diffs = [diff_table[selected_indices[0], k] for k in range(n_points)]
    
    # 生成公式表达式
    formula_terms = [f"f[{selected_x[0]:.1f}]"]
    for k in range(1, n_points):
        term = f"f[{','.join([f'{xi:.1f}' for xi in selected_x[:k+1]])}]"
        product = " ".join([f"(x{'+' if -xi >=0 else ''}{-xi})" for xi in selected_x[:k]])
        formula_terms.append(f"{term} {product}")
    
    formula = "P_{" + f"{n_points-1}" + "}(" + f"{x:.2f}" + ") = " + " + ".join(formula_terms)
    
    # 生成数值代入过程
    numeric_steps = []
    value = used_diffs[0]
    for k in range(1, n_points):
        numer = np.prod([x - xi for xi in selected_x[:k]])
        term_value = used_diffs[k] * numer
        value += term_value
        numeric_steps.append(f"{used_diffs[k]:.4f} \\cdot {numer:.4f} = {term_value:.6f}")
    
    # 生成误差项
    if n_points < len(x_list):
        # 计算乘积项 (x-x0)(x-x1)...(x-xn)
        product = np.prod([x - xi for xi in selected_x])
        # 使用下一个未使用的差商作为误差估计
        next_diff = diff_table[selected_indices[0], n_points]
        error_term = f"f[{','.join([f'{xi:.1f}' for xi in selected_x])}] \\cdot " + \
                     " ".join([f"(x{'+' if -xi >=0 else ''}{-xi})" for xi in selected_x])
        error_value = f"{next_diff:.4f} \\cdot {product:.4f} = {next_diff * product:.6f}"
    else:
        error_term = "无法估计"
        error_value = "无"

    # 组合最终表达式
    latex = f"""
\\subsection*{{{n_points}点牛顿插值}}
使用节点 $x_{{{selected_indices[0]}}}={selected_x[0]:.1f}, ..., x_{{{selected_indices[-1]}}}={selected_x[-1]:.1f}$，构造插值多项式：
\\[
{formula}
\\]

代入数据进行计算：
\\[
\\begin{{aligned}}
P_{{{n_points-1}}}({x:.2f}) &= {" + ".join(numeric_steps)} \\\\
&= {value:.6f}
\\end{{aligned}}
\\]

\\textbf{{截断误差估计}}：
\\[
R_{{{n_points-1}}}(x) \\approx {error_term} = {error_value}
\\]
"""
    return latex

##################### 主程序示例 #########################
x_list = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
y_list = np.array([1.0000, 1.2214, 1.4918, 1.8221, 2.2255])

# 生成插商表
diff_table = generate_divided_diff_table(x_list, y_list)

# 输出插商表LaTeX
diff_table_latex = format_divided_diff_table(x_list, diff_table)
print("插商表LaTeX代码：\n", diff_table_latex)

# 生成三点插值解答
latex_3point = generate_latex_solution_divided(0.12, x_list, diff_table, 3)
print("\n三点牛顿插值解答LaTeX代码：\n", latex_3point)

# 生成四点插值解答
latex_4point = generate_latex_solution_divided(0.12, x_list, diff_table, 4)  # 修改为4
print("\n四点牛顿插值解答LaTeX代码：\n", latex_4point)