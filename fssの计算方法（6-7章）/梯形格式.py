import math

def trapezoidal_method(h, x_end):
    """
    使用梯形格式求解微分方程 y' + y = 0, y(0)=1
    
    参数:
    h : 步长
    x_end : 求解区间的右端点
    
    返回:
    x_list : x值列表
    y_list : 数值解列表
    """
    # 初始条件
    x = x0 = 0.0#修改初值
    y0 = 1.0
    
    # 计算迭代次数
    n = int(x_end / h) + 1
    
    # 初始化列表
    x_list = [x0]
    y_list = [y0]
    
    # 计算比例因子
    factor = (1 - h/2) / (1 + h/2)#修改函数，形式为y_i+1-h/2*f(x_i+1,y_i+1)=y+h/2*f(x_i,y_i)
    
    # 迭代计算
    for i in range(1, n):
        x = x + h
        y = y_list[-1] * factor
        x_list.append(x)
        y_list.append(y)
    
    return x_list, y_list

# 设置参数
h = 0.1
x_end = 1.5

# 计算梯形格式的解
x_list, y_trap = trapezoidal_method(h, x_end)

# 计算精确解
y_exact = [math.exp(-x) for x in x_list]#修改精确解

# 计算误差
errors = [abs(y_trap[i] - y_exact[i]) for i in range(len(x_list))]

# 输出结果
print("梯形格式求解 y' + y = 0, y(0)=1 (h=0.1)")
print("=" * 60)
print(f"{'x':<8}{'梯形格式':<15}{'精确解':<15}{'绝对误差':<15}")
print("-" * 60)
for i in range(len(x_list)):
    print(f"{x_list[i]:<8.1f}{y_trap[i]:<15.7f}{y_exact[i]:<15.7f}{errors[i]:<15.2e}")
print("=" * 60)

# 输出最大误差
max_error = max(abs(err) for err in errors)
print(f"最大绝对误差: {max_error:.2e}")