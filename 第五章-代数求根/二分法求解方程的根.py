import math
def bisection_method(f, a, b, tol, max_iter=100):
    """
    使用二分法求解方程f(x)=0在区间[a, b]内的根
    
    参数:
    f (function): 目标函数
    a (float): 初始区间左端点
    b (float): 初始区间右端点
    tol (float): 允许误差（精确到小数点后第五位对应tol=0.5e-5）
    max_iter (int): 最大迭代次数（防止无限循环）
    
    返回:
    float: 近似根
    int: 实际迭代次数
    list: 迭代过程记录（迭代次数，a, b, mid, f(mid)）
    """
    # 检查初始区间是否有效
    if f(a) * f(b) >= 0:
        raise ValueError("区间端点函数值必须异号，确保存在根")
    
    iterations = 0
    history = []
    
    while (b - a)/2 > tol and iterations < max_iter:
        mid = (a + b) / 2.0
        f_mid = f(mid)
        
        # 记录当前迭代状态
        history.append((iterations, a, b, mid, f_mid))
        
        # 更新区间
        if f_mid == 0:  # 精确找到根（实际计算中几乎不可能）
            break
        elif f(a) * f_mid < 0:
            b = mid
        else:
            a = mid
        
        iterations += 1
    
    # 计算最终结果并记录
    final_mid = (a + b)/2.0
    history.append((iterations, a, b, final_mid, f(final_mid)))
    
    return final_mid, iterations, history

# 定义目标函数
f = lambda x: 3*x**2-math.e**x

# 设置求解参数
a_initial = 3
b_initial = 4
tolerance = 0.5e-5  # 精确到小数点后第五位

# 执行二分法求解
try:
    root, num_iters, history = bisection_method(f, a_initial, b_initial, tolerance)
    
    # 格式化输出结果
    print(f"{'迭代次数':<8} {'左端点':<12} {'右端点':<12} {'中点':<12} {'f(中点)':<12}")
    for entry in history:
        print(f"{entry[0]:<8} {entry[1]:<12.8f} {entry[2]:<12.8f} {entry[3]:<12.8f} {entry[4]:<12.8f}")
    
    print(f"\n在{num_iters}次迭代后找到根:")
    print(f"x ≈ {root:.8f}")
    print(f"精确到小数点后第五位: {root:.5f}")

except ValueError as e:
    print(e)