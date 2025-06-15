import numpy as np
from sympy import symbols, And, simplify, latex, Rational, Matrix, linsolve, factor, fraction, expand, lambdify
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

def generate_spline_pieces(x_list, y_list, M, boundary_type):
    """构造分段样条函数并输出详细过程"""
    n = len(x_list) - 1
    h = x_list[1] - x_list[0]
    x = symbols('x')
    pieces = []
    
    print("\n【分段多项式构造】")
    print("-" * 60)
    
    for i in range(n):
        xi = x_list[i]
        xi1 = x_list[i+1]
        t = x - xi  # 局部变量替换
        
        # 构造四项
        term1 = (M[i+1]/(6*h)) * t**3
        term2 = (M[i]/(6*h)) * (h - t)**3
        term3 = (y_list[i] - (M[i]*h**2)/6) * (h - t)/h
        term4 = (y_list[i+1] - (M[i+1]*h**2)/6) * t/h
        
        # 展示构造过程
        print(f"\n区间 [{xi}, {xi1}] 构造步骤:")
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
    x_vals = np.linspace(min(x_list) - 0.5, max(x_list) + 0.5, 400)
    y_vals = np.zeros_like(x_vals)
    
    # 计算样条值
    for i, (cond, expr) in enumerate(pieces):
        func = lambdify(x, expr, 'numpy')
        mask = np.vectorize(lambda val: bool(cond.subs(x, val)))(x_vals)
        y_vals[mask] = func(x_vals[mask])
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label="三次样条", color='blue')
    plt.scatter(x_list, y_list, color='red', label="原始数据点")
    
    # 添加坐标标注
    for xi, yi in zip(x_list, y_list):
        plt.annotate(f"({xi:.1f}, {yi:.2f})", (xi, yi), 
                    textcoords="offset points", xytext=(0, 10), 
                    ha='center')
    
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("三次样条插值曲线")
    plt.legend()
    plt.grid(True)
    plt.show()

def cubic_spline_generator(x_list, y_list, boundary_type, boundary_values):
    """主函数：生成三次样条插值"""
    n = len(x_list) - 1
    h = x_list[1] - x_list[0]
    
    print("【输入数据】")
    print(f"x_list = {x_list}")
    print(f"y_list = {y_list}")
    print(f"边界条件: {boundary_type}")
    print("-" * 60)
    
    # 构造三弯矩方程
    A = np.zeros((n+1, n+1), dtype=object)
    d = np.zeros(n+1, dtype=object)
    
    # 内部节点方程
    for i in range(1, n):
        A[i, i-1] = Rational(1,6)
        A[i, i] = Rational(2,3)
        A[i, i+1] = Rational(1,6)
        d[i] = Rational((y_list[i+1] - 2*y_list[i] + y_list[i-1]), h**2)
    
    # 边界条件处理
    if boundary_type == 'natural':
        A[0, 0] = 1
        A[-1, -1] = 1
        d[0] = Rational(boundary_values[0])
        d[-1] = Rational(boundary_values[1])
    elif boundary_type == 'clamped':
        A[0, 0] = Rational(2*h,3)
        A[0, 1] = Rational(h,6)
        d[0] = Rational((y_list[1]-y_list[0])/h - boundary_values[0])
        
        A[-1, -2] = Rational(h,6)
        A[-1, -1] = Rational(2*h,3)
        d[-1] = Rational(boundary_values[1] - (y_list[-1]-y_list[-2])/h)
    
    # 求解方程
    M = solve_moment_equations(A, d, n)
    
    # 构造分段函数
    pieces, x = generate_spline_pieces(x_list, y_list, M, boundary_type)
    
    # 生成LaTeX表达式
    print("\n【最终分段函数】")
    print("-" * 60)
    latex_str = "S(x) = \\begin{cases}\n"
    for cond, expr in pieces:
        expr_str = latex(expr, fold_func_brackets=True, mul_symbol='times')
        expr_str = expr_str.replace(r'\left', '').replace(r'\right', '')
        cond_str = latex(cond).replace('\\wedge', '且').replace('&', '')
        latex_str += f"{expr_str}, & {cond_str} \\\\\n"
    latex_str += "\\end{cases}"
    
    print(latex_str)
    
    # 可视化
    plot_spline(x_list, y_list, pieces, x)
    
    return pieces

# 示例调用
if __name__ == "__main__":
    # 示例1：自然边界条件
    print("\n【示例1：自然边界条件】")
    x_list = [0, 1, 2, 3]
    y_list = [0, 0, 0, 0]
    cubic_spline_generator(x_list, y_list, 'natural', [1, 0])
    
    # 示例2：夹持边界条件
    print("\n【示例2：夹持边界条件】")
    x_list = [0, 1, 2, 3]
    y_list = [0, 0, 0, 0]
    cubic_spline_generator(x_list, y_list, 'clamped', [1, 0])