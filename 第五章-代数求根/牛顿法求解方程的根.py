import math

def newton_method(f, df, f_str, df_str, x0, tol, max_iter):
    """boatchanting

    牛顿法求解方程 f(x) = 0 的近似根

    参数:
    f        : 目标函数 (Python 函数)

    df       : 目标函数的导数 (Python 函数)

    f_str    : f(x) 的字符串表示 (用于输出)

    df_str   : f'(x) 的字符串表示 (用于输出)

    x0       : 初始猜测值

    tol      : 精度要求 (|x_{k+1} - x_k| < tol)

    max_iter : 最大迭代次数

    示例使用：
    >>> def f(x):
    ...     return x**2 - 4
    >>> def df(x):
    ...     return 2*x
    >>> newton_method(f, df, "x^2 - 4", "2x", 1.5, 1e-5, 100)

    """
    # 打印函数定义
    print("【函数及其导数】")
    print(f"  定义函数：{f_str}")
    print(f"  计算导数：{df_str}\n")

    # 打印迭代公式
    print("【牛顿迭代公式】")
    print(f"  迭代公式为：x_{max_iter} = x_k - f(x_k)/f'(x_k)\n")

    # 初始化迭代数据
    iter_data = []
    xk = x0
    for i in range(max_iter):
        fxk = f(xk)
        dfxk = df(xk)

        # 防止除以零
        if dfxk == 0:
            print("【错误】导数为零，无法继续迭代。")
            return

        # 下一次迭代值
        xk_next = xk - fxk / dfxk
        error = abs(xk_next - xk)

        # 保存迭代数据
        iter_data.append((i, xk, fxk, dfxk, xk_next, error))

        # 判断是否满足精度要求
        if error < tol:
            break

        xk = xk_next

    # 打印迭代过程表格
    print("【迭代过程】")
    print("  迭代结果如下表所示（保留6位小数）：")
    print("  ┌──────────┬────────────┬────────────┬────────────┬────────────┬──────────────┐")
    print("  │ 迭代次数 │   x_k       │  f(x_k)     │ f'(x_k)    │ x_{k+1}     │ |x_{k+1}-x_k|│")
    print("  ├──────────┼────────────┼────────────┼────────────┼────────────┼──────────────┤")
    for data in iter_data:
        print(f"  │ {data[0]:8} │ {data[1]:10.6f} │ {data[2]:10.6f} │ {data[3]:10.6f} │ {data[4]:10.6f} │ {data[5]:12.6f} │")
    print("  └──────────┴────────────┴────────────┴────────────┴────────────┴──────────────┘\n")

    # 打印最终结果
    print("【最终结果】")
    if error < tol:
        print(f"  满足精度要求的近似根为：{xk_next:.6f}")
    else:
        print(f"  超过最大迭代次数，未达到精度要求。当前近似根为：{xk_next:.6f}")

def f1(x):
    return x**2 + 10 * math.cos(x)

def df1(x):
    return 2 * x - 10 * math.sin(x)

# 调用牛顿法
newton_method(
    f=f1,
    df=df1,
    f_str="f(x) = x^2 + 10*cos(x)",
    df_str="f'(x) = 2x - 10*sin(x)",
    x0=1.6,
    tol=1e-5,
    max_iter=100
)

# 定义目标函数
def f2(x):
    return 1 + math.atan(x) - x

# 定义导数函数
def df2(x):
    return 1 / (1 + x**2) - 1

# 调用牛顿法求根
newton_method(
    f=f2,
    df=df2,
    f_str="f(x) = 1 + arctan(x) - x",
    df_str="f'(x) = 1/(1 + x^2) - 1",
    x0=2,         # 初始猜测值（根据函数特性选择）
    tol=1e-5,       # 精度要求
    max_iter=100    # 最大迭代次数
)

import math

# 定义目标函数
def f(x):
    return x**2 - 30

# 定义导数函数
def df(x):
    return 2 * x

# 调用牛顿法求根
newton_method(
    f=f,
    df=df,
    f_str="f(x) = x^2 - 30",
    df_str="f'(x) = 2x",
    x0=5.0,         # 初始猜测值（接近√30 ≈ 5.477）
    tol=1e-5,       # 精度要求
    max_iter=100    # 最大迭代次数
)