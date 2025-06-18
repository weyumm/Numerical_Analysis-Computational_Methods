import numpy as np
from sympy import symbols, And, simplify, latex, Rational, Matrix, linsolve, factor, fraction

def format_fraction(expr):
    expr = expr.expand()
    num, den = fraction(expr)
    return Rational(num, den) if den != 1 else expr

def cubic_spline_generator(x_list, y_list, boundary_type, boundary_values):
    n = len(x_list) - 1
    h = x_list[1] - x_list[0]   #无法解决步长不相等的问题
    
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
    
    # 修改符号求解后的代码段
    # 符号求解
    A_sym = Matrix(A.tolist())
    d_sym = Matrix(d)
    M = list(linsolve((A_sym, d_sym)))[0]
    
    # 添加三弯矩方程输出
    print("\n三弯矩方程:")
    for i in range(len(A)):
        eq_str = " + ".join([f"{latex(A[i,j])}M_{j}" for j in range(len(A)) if A[i,j] != 0]) + f" = {latex(d[i])}"
        print(f"方程 {i+1}: {eq_str}")

    # 构造分段函数
    x = symbols('x')
    pieces = []
    for i in range(n):
        xi = x_list[i]
        t = symbols(f't_{i}')
        term1 = (M[i+1]/(6*h)) * t**3
        term2 = (M[i]/(6*h)) * (h - t)**3
        term3 = (y_list[i] - (M[i]*h**2)/6) * (h - t)/h
        term4 = (y_list[i+1] - (M[i+1]*h**2)/6) * t/h
        
        # 添加多项式构造过程输出
        print(f"\n区间 [{xi}, {x_list[i+1]}] 构造步骤:")
        print(f"项1 (弯矩影响): {latex(term1.subs(t, x-xi))}")
        print(f"项2 (反向弯矩): {latex(term2.subs(t, x-xi))}") 
        print(f"项3 (线性修正): {latex(term3.subs(t, x-xi))}")
        print(f"项4 (端点修正): {latex(term4.subs(t, x-xi))}")
        
        S_i = simplify(term1 + term2 + term3 + term4)
        S_i = factor(format_fraction(S_i.subs(t, x - xi)))
        condition = (xi <= x) & (x < x_list[i+1])
        pieces.append( (condition, S_i) )
        print(f"组合结果: {latex(S_i)}")
    
    # 生成LaTeX（禁用\left和\right）
    latex_str = "S(x) = \\begin{cases}\n"
    for cond, expr in pieces:
        expr_str = latex(expr, fold_func_brackets=True, mul_symbol='times')
        expr_str = expr_str.replace(r'\left', '').replace(r'\right', '')  # 关键修改
        expr_str = expr_str.replace('(x - 0)', 'x')  # 优化显示
        cond_str = latex(cond).replace('\\wedge', '且').replace('&', '')
        latex_str += f"{expr_str}, & {cond_str} \\\\\n"
    latex_str += "\\end{cases}"
    return latex_str

# 测试用例
x_list = [-1.5, 0, 1, 2]
y_list = [0.125, -1, 1, 9]

print("问题(1) 自然边界条件：")# 边界二阶导数值（左，右）
print(cubic_spline_generator(x_list, y_list, 'natural', [0, 0]))

print("\n问题(2) 固定一阶导数边界条件：")# 边界一阶导数值（左，右）
print(cubic_spline_generator(x_list, y_list, 'clamped', [0.75,14]))