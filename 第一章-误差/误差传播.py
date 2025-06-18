import sympy as sp
import math

def error_propagation(expr_str, variables, values, uncertainties):
    """
    计算函数值的误差传播
    
    参数:
    expr_str: 字符串形式的数学表达式 (例如 "x*y + z**2")
    variables: 变量名称列表 (例如 ['x','y','z'])
    values: 变量测量值列表 (例如 [1.0, 2.0, 3.0])
    uncertainties: 变量不确定度列表 (例如 [0.1, 0.05, 0.01])
    
    返回:
    (函数值, 总不确定度)
    """
    # 创建符号变量
    symbols = sp.symbols(' '.join(variables))
    sym_dict = dict(zip(variables, symbols))
    
    # 解析表达式
    expr = sp.sympify(expr_str, locals=sym_dict)
    
    # 计算函数值
    value_dict = dict(zip(symbols, values))
    f_value = float(expr.evalf(subs=value_dict))
    
    # 计算偏导数和不确定度分量
    sum_squares = 0.0
    for var, val, unc in zip(symbols, values, uncertainties):
        # 计算偏导数
        partial = sp.diff(expr, var)
        partial_val = float(partial.evalf(subs=value_dict))
        
        # 计算不确定度分量并累加
        sum_squares += (partial_val * unc) ** 2
    
    # 计算总不确定度
    total_uncertainty = math.sqrt(sum_squares)
    
    return f_value, total_uncertainty

# 示例使用
if __name__ == "__main__":
    # 定义表达式和参数
    expression = "x*y + z**2 / w"
    vars = ['x', 'y', 'z', 'w']
    vals = [2.0, 3.0, 4.0, 5.0]
    uncs = [0.1, 0.15, 0.2, 0.1]
    
    # 计算误差传播
    f_val, uncertainty = error_propagation(expression, vars, vals, uncs)
    
    # 打印结果
    print(f"函数表达式: {expression}")
    print(f"变量值: {dict(zip(vars, vals))}")
    print(f"测量值: f = {f_val:.4f}")
    print(f"总不确定度: σ_f = {uncertainty:.4f}")
    print(f"最终结果: f = ({f_val:.4f} ± {uncertainty:.4f})")