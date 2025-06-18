import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def runge_kutta_method(f, x0, y0, h, xn):
    """
    四阶龙格-库塔法求解微分方程
    
    参数:
    f : 微分方程 dy/dx = f(x, y)
    x0, y0 : 初始条件
    h : 步长
    xn : 终点
    """
    x = np.arange(x0, xn + h, h)
    y = np.zeros(len(x))
    y[0] = y0

    for i in range(1, len(x)):
        k1 = h * f(x[i-1], y[i-1])
        k2 = h * f(x[i-1] + h/2, y[i-1] + k1/2)
        k3 = h * f(x[i-1] + h/2, y[i-1] + k2/2)
        k4 = h * f(x[i-1] + h, y[i-1] + k3)
        y[i] = y[i-1] + (k1 + 2*k2 + 2*k3 + k4)/6

    return x, y

def plot_runge_kutta(x, y):
    """
    可视化龙格-库塔法的数值解结果
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制数值解
    ax.plot(x, y, 'o-', label='数值解', color='blue', markersize=6, linewidth=1.5)
    
    ax.set_title("四阶龙格-库塔法数值解", fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("y(x)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()

# 示例微分方程：dy/dx = -2/(y - x), y(0)=1
def f(x, y):
    return -2.0 / (y - x)

# 参数设置
x0 = 0.0  # 初始点的x值，微分方程的初始条件y(0)=1中的x=0
y0 = 1.0  # 初始点的y值，微分方程的初始条件y(0)=1中的y=1
h = 0.1 #步长（每一步x的增量），将区间[0,1]分成 10 等分，每步计算一次近似解
xn = 1.0  # 终止点的x值，微分方程求解的终止点

# 执行计算
x_values, y_values = runge_kutta_method(f, x0, y0, h, xn)

# 输出结果表格
print("i\tx_i\t计算值y_i")
print("-" * 40)
for i in range(len(x_values)):
    print(f"{i}\t{x_values[i]:.6f}\t{y_values[i]:.6f}")

# 可视化结果
plot_runge_kutta(x_values, y_values)