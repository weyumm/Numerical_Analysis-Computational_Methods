import numpy as np
import matplotlib.pyplot as plt
import math
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
def f(x):
    """默认被积函数（可修改）"""
    return 4 / (1 + x**2)  # 示例函数

def romberg_integration(f, a, b, tol=0.5e-6, max_iter=10):
    """
    龙贝格积分算法实现
    参数:
        f: 被积函数
        a: 积分下限
        b: 积分上限
        tol: 容差（精度要求）
        max_iter: 最大迭代次数
    返回:
        数值积分结果
    """
    R = np.zeros((max_iter, max_iter))
    print("开始龙贝格积分过程...\n")

    # 初始梯形公式
    R[0, 0] = 0.5 * (b - a) * (f(a) + f(b))
    print(f"第 0 步 R[0,0] = {R[0, 0]:.8f}")
    print_table(R, 0)

    for n in range(1, max_iter):
        h = (b - a) / (2 ** n)
        sum_f = sum(f(a + (2 * k - 1) * h) for k in range(1, 2 ** (n - 1) + 1))
        R[n, 0] = 0.5 * R[n - 1, 0] + h * sum_f  # 梯形规则细分

        print(f"第 {n} 步 R[{n},0] = {R[n, 0]:.8f}")

        # Richardson 外推
        for m in range(1, n + 1):
            R[n, m] = R[n, m - 1] + (R[n, m - 1] - R[n - 1, m - 1]) / (4 ** m - 1)
            print(f"       R[{n},{m}] = {R[n, m]:.8f}")

        print_table(R, n)

        # 收敛判断
        if abs(R[n, n] - R[n - 1, n - 1]) < tol:
            print(f"\n积分结果为：{R[n, n]:.8f}")
            print(f"使用 {n+1} 次迭代达到精度要求")
            return R[n, n], R[:n+1, :n+1]  # 返回积分结果和中间表

    raise ValueError("Romberg 积分在最大迭代次数内未收敛")

# 打印当前 R 表格
def print_table(R, current_row):
    """格式化打印Romberg表"""
    print("当前 Romberg 表：")
    header = [" "] + [f"m={i}" for i in range(current_row+1)]
    print("{:<8}".format(header[0]), end="")
    for col in header[1:]:
        print("{:<12}".format(col), end="")
    print()

    for i in range(current_row+1):
        print("{:<8}".format(f"n={i}"), end="")
        for j in range(i+1):
            val = f"{R[i, j]:.6f}"
            print("{:<12}".format(val), end="")
        print()
    print()

def plot_function(f, a, b, result):
    """
    绘制被积函数图像
    参数:
        f: 被积函数
        a: 积分下限
        b: 积分上限
        result: 积分结果
    """
    x = np.linspace(a, b, 400)
    y = [f(xi) for xi in x]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', label='被积函数 f(x)')
    plt.fill_between(x, y, alpha=0.2, color='gray', label=f'积分区域 (面积 ≈ {result:.6f})')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('龙贝格积分可视化')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_convergence(R):
    """
    绘制积分收敛过程
    参数:
        R: Romberg表
    """
    steps = range(len(R))
    values = [R[i, i] for i in steps]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, values, 'bo-', label='Romberg 估计值')
    plt.axhline(y=values[-1], color='r', linestyle='--', label=f'最终结果 = {values[-1]:.6f}')
    plt.xlabel('迭代步数')
    plt.ylabel('积分估计值')
    plt.title('积分收敛过程')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    """
    主函数：用户可修改参数部分
    """
    # ==================================
    # 用户可配置参数区域
    # ==================================
    # 1. 定义被积函数（支持math库函数）
    def integrand(x):
        return math.sin(x)  # 可修改为任意可积函数
    
    # 2. 设置积分区间
    a = 0         # 积分下限
    b = math.pi   # 积分上限
    
    # 3. 设置精度要求
    tolerance = 1e-8  # 收敛容差
    
    # 4. 设置最大迭代次数
    max_iterations = 15  # 最大递归次数
    
    # 5. 是否显示详细过程
    show_details = True  # 设置为False可关闭详细输出
    
    # ==================================
    # 程序执行区域
    # ==================================
    print("正在执行龙贝格积分...")
    
    try:
        # 执行积分计算
        result, R_table = romberg_integration(
            f=integrand,
            a=a,
            b=b,
            tol=tolerance,
            max_iter=max_iterations
        )
        
        print(f"\n最终积分结果：{result:.10f}")
        
        # 可视化函数图像
        plot_function(integrand, a, b, result)
        
        # 绘制收敛过程
        plot_convergence(R_table)
        
    except ValueError as e:
        print(f"积分计算异常：{e}")