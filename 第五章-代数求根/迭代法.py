import numpy as np
import matplotlib.pyplot as plt

# 定义函数
def f(x):
    return 3*x**2 - np.exp(x)

# 简单迭代法函数
def simple_iteration(g, x0, tol=1e-6, max_iter=100, root_name=""):
    print(f"\n求解 {root_name} 根:")
    print(f"{'迭代次数':<8}{'x':<15}{'g(x)':<15}{'误差':<15}")
    print("-" * 45)
    
    x = x0
    history = [x]  # 保存迭代历史
    
    for i in range(max_iter):
        x_new = g(x)
        error = abs(x_new - x)
        
        # 打印当前迭代信息
        print(f"{i+1:<8}{x:<15.8f}{x_new:<15.8f}{error:<15.8f}")
        
        # 检查收敛条件
        if error < tol:
            print(f"\n在 {i+1} 次迭代后收敛到根: {x_new:.8f}")
            return x_new, history
        
        x = x_new
        history.append(x)
    
    print(f"\n警告: 在 {max_iter} 次迭代内未达到收敛条件")
    return x, history

# 可视化函数和迭代过程
def plot_function_and_iteration(f, roots, histories, x_range=(-2, 4)):
    plt.figure(figsize=(12, 8))
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'SimHei'
    
    # 绘制函数曲线
    x = np.linspace(x_range[0], x_range[1], 400)
    y = f(x)
    plt.plot(x, y, 'b-', linewidth=2, label=r'$f(x) = 3x^2 - e^x$')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 标记根
    colors = ['red', 'green', 'purple']
    for i, root in enumerate(roots):
        color = colors[i % len(colors)]
        history = histories[i]
        
        # 绘制迭代路径
        iter_x = np.array(history)
        iter_y = f(iter_x)
        plt.plot(iter_x, iter_y, 'o-', color=color, markersize=6, 
                 label=f'{root[2]}根迭代路径')
        
        # 标记最终根
        plt.plot(root[0], f(root[0]), 's', markersize=8, color=color, 
                 markerfacecolor='white', markeredgewidth=2)
        plt.annotate(f'{root[2]}根: {root[0]:.6f}', (root[0], f(root[0])), 
                     xytext=(root[0]+0.5, f(root[0])+1), 
                     arrowprops=dict(arrowstyle="->", color=color))
    
    plt.title('简单迭代法求解 $3x^2 - e^x = 0$', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    # 方程有三个实根: 
    # 负根 ≈ -0.45, 中间根 ≈ 0.91, 正根 ≈ 3.73
    
    # 1. 求解负根 (x ≈ -0.45)
    # 迭代函数: g(x) = -sqrt(e^x/3)
    g_negative = lambda x: -np.sqrt(np.exp(x)/3)
    neg_root, neg_history = simple_iteration(g_negative, -0.5, root_name="负")
    
    # 2. 求解中间根 (x ≈ 0.91)
    # 迭代函数: g(x) = exp(x/2)/sqrt(3)
    g_middle = lambda x: np.exp(x/2) / np.sqrt(3)
    mid_root, mid_history = simple_iteration(g_middle, 0.5, root_name="中间")
    
    # 3. 求解正根 (x ≈ 3.73)
    # 迭代函数: g(x) = sqrt(e^x/3)
    g_positive = lambda x: np.sqrt(np.exp(x)/3)
    pos_root, pos_history = simple_iteration(g_positive, 3.5, root_name="正")
    
    # 收集结果
    roots = [
        (neg_root, neg_history, "负"),
        (mid_root, mid_history, "中间"),
        (pos_root, pos_history, "正")
    ]
    
    # 可视化结果
    plot_function_and_iteration(f, roots, histories=[h for _, h, _ in roots])
    
    # 最终结果汇总
    print("\n" + "="*50)
    print("方程 3x² - eˣ = 0 的根:")
    print("="*50)
    print(f"负根: x ≈ {neg_root:.8f}")
    print(f"中间根: x ≈ {mid_root:.8f}")
    print(f"正根: x ≈ {pos_root:.8f}")
    print("="*50)