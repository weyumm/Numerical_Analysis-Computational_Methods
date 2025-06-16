import numpy as np
from sympy import symbols, And, simplify, latex, Rational, Matrix, linsolve, fraction, expand, lambdify
import matplotlib.pyplot as plt

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def format_fraction(expr):
    """格式化分数表达式"""
    expr = expr.expand()
    num, den = fraction(expr)
    return Rational(num, den) if den != 1 else expr

def solve_moment_equations(A, d, n):
    """求解三弯矩方程并输出详细过程"""
    print("\n【三弯矩方程组】")
    print("-" * 60)
    
    # 打印方程系数矩阵
    print("系数矩阵A:")
    for row in A:
        print(["%.4f" % float(x) for x in row])
    
    print("\n常数项d:")
    print(["%.4f" % float(x) for x in d])
    
    # 符号求解
    A_sym = Matrix(A.tolist())
    d_sym = Matrix(d)
    solution = list(linsolve((A_sym, d_sym)))[0]
    
    print("\n【方程组解】")
    print("-" * 60)
    for i, val in enumerate(solution):
        print(f"M_{i} = {simplify(val)}")
    
    return solution

def generate_spline_pieces(x_list, y_list, M, h_list):
    """构造分段样条函数并输出详细过程"""
    n = len(x_list) - 1
    x = symbols('x')
    pieces = []
    
    print("\n【分段多项式构造】")
    print("-" * 60)
    
    for i in range(n):
        xi = x_list[i]
        xi1 = x_list[i+1]
        h_i = h_list[i]
        t = x - xi  # 局部变量替换
        
        # 使用当前区间的步长h_i
        term1 = (M[i+1]/(6*h_i)) * t**3
        term2 = (M[i]/(6*h_i)) * (h_i - t)**3
        # 修正端点值计算
        term3 = (y_list[i] - (M[i]*h_i**2)/6) * (h_i - t)/h_i
        term4 = (y_list[i+1] - (M[i+1]*h_i**2)/6) * t/h_i
        
        # 展示构造过程
        print(f"\n区间 [{xi}, {xi1}] 构造步骤 (h={h_i:.2f}):")
        print(f"项1 (弯矩影响): {expand(term1)}")
        print(f"项2 (反向弯矩): {expand(term2)}")
        print(f"项3 (线性修正): {expand(term3)}")
        print(f"项4 (端点修正): {expand(term4)}")
        
        # 合并同类项
        S_i = simplify(term1 + term2 + term3 + term4)
        S_i_expanded = expand(S_i)
        condition = And(xi <= x, x < xi1)
        pieces.append((condition, S_i_expanded))
        
        print(f"组合结果: {S_i_expanded}")
    
    return pieces, x

def plot_spline(x_list, y_list, pieces, x):
    """绘制样条曲线和原始数据点"""
    print("\n【可视化展示】")
    print("-" * 60)
    
    # 生成绘图数据
    x_min, x_max = min(x_list), max(x_list)
    padding = 0.1 * (x_max - x_min)
    x_vals = np.linspace(x_min - padding, x_max + padding, 500)
    y_vals = np.zeros_like(x_vals)
    
    # 计算样条值
    for i, (cond, expr) in enumerate(pieces):
        func = lambdify(x, expr, 'numpy')
        mask = np.vectorize(lambda val: bool(cond.subs(x, val)))(x_vals)
        y_vals[mask] = func(x_vals[mask])
    
    # 绘图
    plt.figure(figsize=(12, 7))
    plt.plot(x_vals, y_vals, label="三次样条", color='blue')
    plt.scatter(x_list, y_list, color='red', s=80, zorder=5, label="原始数据点")
    
    # 添加节点垂直线
    for xi in x_list:
        plt.axvline(x=xi, color='gray', linestyle='--', alpha=0.5)
    
    # 添加坐标标注（智能位置）
    for i, (xi, yi) in enumerate(zip(x_list, y_list)):
        offset = 15 if i % 2 == 0 else -25  # 交替显示在上方/下方
        plt.annotate(f"({xi:.1f}, {yi:.2f})", (xi, yi), 
                    textcoords="offset points", xytext=(0, offset), 
                    ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel("x", fontsize=12)
    plt.ylabel("f(x)", fontsize=12)
    plt.title("三次样条插值曲线", fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def cubic_spline_generator(x_list, y_list, boundary_type, boundary_values):
    """主函数：生成三次样条插值"""
    n = len(x_list) - 1
    # 计算每个区间的步长
    h_list = [x_list[i+1] - x_list[i] for i in range(n)]
    
    print("【输入数据】")
    print(f"x_list = {x_list}")
    print(f"y_list = {y_list}")
    print(f"区间步长: {h_list}")
    print(f"边界条件: {boundary_type}")
    print("-" * 60)
    
    # 构造三弯矩方程
    A = np.zeros((n+1, n+1), dtype=object)
    d = np.zeros(n+1, dtype=object)
    
    # 内部节点方程 (i=1 to n-1)
    for i in range(1, n):
        h_prev = h_list[i-1]
        h_curr = h_list[i]
        total_h = h_prev + h_curr
        
        A[i, i-1] = h_prev / total_h
        A[i, i] = 2
        A[i, i+1] = h_curr / total_h
        
        # 计算右侧差分项
        diff1 = (y_list[i] - y_list[i-1]) / h_prev
        diff2 = (y_list[i+1] - y_list[i]) / h_curr
        d[i] = 6 * (diff2 - diff1) / total_h
    
    # 边界条件处理
    if boundary_type == 'natural':
        # 自然边界：固定M0=0和Mn=0
        A[0, 0] = 1
        d[0] = 0
        
        A[n, n] = 1
        d[n] = 0
        print("自然边界条件：M₀ = 0, Mₙ = 0")
    elif boundary_type == 'clamped':
        # 夹持边界：使用用户提供的一阶导数值
        h0 = h_list[0]
        A[0, 0] = 2 * h0
        A[0, 1] = h0
        d[0] = 6 * ((y_list[1] - y_list[0]) / h0 - boundary_values[0])
        
        hn = h_list[-1]
        A[n, n-1] = hn
        A[n, n] = 2 * hn
        d[n] = 6 * (boundary_values[1] - (y_list[n] - y_list[n-1]) / hn)
        print(f"夹持边界条件：f'({x_list[0]}) = {boundary_values[0]}, f'({x_list[-1]}) = {boundary_values[1]}")
    
    # 求解方程
    M = solve_moment_equations(A, d, n)
    
    # 构造分段函数（传入步长列表）
    pieces, x = generate_spline_pieces(x_list, y_list, M, h_list)
    
    # 生成LaTeX表达式
    print("\n【最终分段函数】")
    print("-" * 60)
    latex_str = "S(x) = \\begin{cases}\n"
    for cond, expr in pieces:
        expr_str = latex(expr, fold_func_brackets=True, mul_symbol='times')
        expr_str = expr_str.replace(r'\left', '').replace(r'\right', '')
        cond_str = latex(cond).replace('\\wedge', '且').replace('&', '')
        latex_str += f"{expr_str} & \\text{{若 }} {cond_str} \\\\\n"
    latex_str += "\\end{cases}"
    
    print(latex_str)
    
    # 可视化
    plot_spline(x_list, y_list, pieces, x)
    
    return pieces

# 示例调用
if __name__ == "__main__":
    # 修正的示例1：自然边界条件（非等距节点）
    print("\n【示例1：自然边界条件（非等距节点）】")
    x_list = [-1.5, 0, 1, 2]
    y_list = [0.125, -1, 1, 9]
    cubic_spline_generator(x_list, y_list, 'natural', [0, 0])  # 自然边界固定M0=Mn=0
    
    # 示例2：夹持边界条件
    print("\n【示例2：夹持边界条件】")
    x_list = [0, 1, 2, 3]
    y_list = [0, 0, 0, 0]
    cubic_spline_generator(x_list, y_list, 'clamped', [1, 0])