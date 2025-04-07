import numpy as np

def generate_backward_diff_table(y_list):
    """生成向后差分表"""
    y = np.asarray(y_list, dtype=float)
    n = len(y)
    diff_table = np.zeros((n, n)) * np.nan
    diff_table[:, 0] = y
    
    for j in range(1, n):
        for i in range(n-1, j-1, -1):
            diff_table[i, j] = diff_table[i, j-1] - diff_table[i-1, j-1]
            
    return diff_table

def format_backward_diff_table(x_list, diff_table):
    """将向后差分表格式化为Markdown表格"""
    header = ["x", "e^x"]
    max_order = diff_table.shape[1] - 1
    header += [f"∇^{i}y" for i in range(1, max_order+1)]
    
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

def select_nodes_backward(x, x_list, n_points):
    """根据x值选择最近的n_points个节点（从后向前）"""
    distances = np.abs(x_list - x)
    sorted_indices = np.argsort(distances)
    # 从距离最近的节点开始，向后取n_points个节点
    selected_indices = []
    for idx in sorted_indices:
        if idx + n_points <= len(x_list):
            selected_indices = list(range(idx, idx + n_points))
            break
    if not selected_indices:
        selected_indices = list(range(len(x_list)-n_points, len(x_list)))
    selected_x = x_list[selected_indices]
    return selected_x, selected_indices

def generate_latex_solution_backward(x, x_list, diff_table, n_points):
    """生成LaTeX解答过程（向后差分）"""
    # 修改节点选择逻辑：根据x的距离选择最近的n_points个点
    selected_x, selected_indices = select_nodes_backward(x, x_list, n_points)
    
    h = selected_x[1] - selected_x[0]  # 使用选中的节点计算步长
    s = (x - selected_x[-1])/h  # 向后差分从最后一个节点开始
    s_round = round(s, 2)
    
    # 获取使用的差分项（从最后一个选中的节点开始）
    used_diffs = diff_table[selected_indices[-1], :n_points][::-1]
    
    # 生成公式表达式
    formula_terms = [f"f(x_n)"]
    for k in range(1, n_points):
        term = f"\\frac{{\\nabla^{k} f(x_n)}}{{{np.math.factorial(k)}}} " + \
               " ".join([f"(s{'+' if i >=0 else ''}{i})" for i in range(k)])
        formula_terms.append(term)
    
    formula = "P_{" + f"{n_points-1}" + "}(" + f"{x:.2f}" + ") = " + " + ".join(formula_terms)
    
    # 生成数值代入过程
    numeric_steps = []
    for k in range(n_points):
        numer = np.prod([s_round + i for i in range(k)])
        denom = np.math.factorial(k)
        term = f"\\frac{{{used_diffs[k]:.4f}}}{{{denom}}} \\cdot {numer:.4f}"
        numeric_steps.append(term)
    
    # 生成误差项
    if n_points < diff_table.shape[1]:
        error_term = f"\\frac{{\\nabla^{n_points} f(x_n)}}{{{np.math.factorial(n_points)}}} " + \
                     " ".join([f"(s{'+' if i >=0 else ''}{i})" for i in range(n_points)])
        error_value = f"\\frac{{{diff_table[-1, n_points]:.4f}}}{{{np.math.factorial(n_points)}}} " + \
                      f"\\cdot {np.prod([s_round + i for i in range(n_points)]):.4f}"
    else:
        error_term = "无法估计"
        error_value = "无"
    
    # 组合最终表达式
    latex = f"""
\\subsection*{{{n_points}点向后插公式}}
使用节点 $x_{{{selected_indices[0]}}}={selected_x[0]:.1f}, ..., x_{{{selected_indices[-1]}}}={selected_x[-1]:.1f}$，构造插值多项式：
\\[
{formula}
\\]
其中步长 $h={h:.1f}$，计算参数 $s = \\frac{{{x:.2f} - {selected_x[-1]:.1f}}}{{{h:.1f}}} = {s_round:.2f}$

代入差分表数据：
\\[
\\begin{{aligned}}
P_{{{n_points-1}}}({x:.2f}) &= {" + ".join(numeric_steps)} \\\\
&= {sum([used_diffs[k]/np.math.factorial(k)*np.prod([s_round + i for i in range(k)]) for k in range(n_points)]):.6f}
\\end{{aligned}}
\\]

\\textbf{{截断误差估计}}：
\\[
R_{{{n_points-1}}} \\approx {error_term} = {error_value} \\approx {diff_table[-1, n_points]/np.math.factorial(n_points)*np.prod([s_round + i for i in range(n_points)]):.2e}
\\]
"""
    return latex

##################### 主程序示例 #########################
x_list = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
y_list = np.array([1.0000, 1.2214, 1.4918, 1.8221, 2.2255])

# 生成向后差分表
diff_table = generate_backward_diff_table(y_list)

# 输出差分表LaTeX
diff_table_latex = format_backward_diff_table(x_list, diff_table)
print("向后差分表LaTeX代码：\n", diff_table_latex)

# 生成三点插值解答
latex_3point = generate_latex_solution_backward(0.7, x_list, diff_table, 3)
print("\n三点向后插值解答LaTeX代码：\n", latex_3point)

# 生成四点插值解答
latex_4point = generate_latex_solution_backward(0.7, x_list, diff_table, 4)
print("\n四点向后插值解答LaTeX代码：\n", latex_4point)