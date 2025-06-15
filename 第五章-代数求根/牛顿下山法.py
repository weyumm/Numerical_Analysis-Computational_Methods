import math

def newton_downhill(
    f, df,
    f_str, df_str,
    x0,
    tol      = 1e-8,
    max_iter = 100,
    alpha    = 0.5,    # 步长衰减系数 0<alpha<1
    lam_min  = 1e-4    # 允许的最小步长
):
    """
    牛顿下山法（带回溯线性搜索）求解 f(x)=0

    参数说明与 classic newton 相同；新增:
    alpha   : 每次回溯时步长 λ ← α·λ，通常 0.1~0.8
    lam_min : 若 λ 减小到 lam_min 仍未下降，则停止并返回
    """

    # ------------------- 打印函数信息 -------------------
    print("【函数及其导数】")
    print(f"  定义函数：{f_str}")
    print(f"  计算导数：{df_str}\n")

    print("【牛顿-下山迭代公式】")
    print("  x_{k+1} = x_k - λ · f(x_k)/f'(x_k)")
    print(f"  其中 λ=1, α={alpha} 做回溯，直至 |f(x_{k+1})| < |f(x_k)|\n")

    # ------------------- 迭代主循环 -------------------
    xk = x0
    iter_data = []

    for k in range(max_iter):
        fxk  = f(xk)
        dfxk = df(xk)

        if dfxk == 0:
            print("【错误】导数为零，无法继续迭代。")
            break

        delta   = -fxk / dfxk        # 纯牛顿步
        lam     = 1.0                # 初始步长
        x_trial = xk + lam * delta   # 试探点

        # ---------- 回溯线性搜索 ----------
        while abs(f(x_trial)) >= abs(fxk):   # 若未下降
            lam *= alpha                     # 缩步长
            if lam < lam_min:                # 步长过小，放弃
                break
            x_trial = xk + lam * delta

        err = abs(x_trial - xk)
        iter_data.append((k, xk, fxk, dfxk, lam, x_trial, err))

        # 收敛判据
        if err < tol:
            xk = x_trial
            break

        # 若步长太小仍未改进，则判为失败
        if lam < lam_min and abs(f(x_trial)) >= abs(fxk):
            print("【警告】步长减到最小仍未下降，算法提前终止。")
            xk = x_trial
            break

        xk = x_trial

    # ------------------- 打印迭代表 -------------------
    print("【迭代过程】")
    print("  ┌──────────┬────────────┬────────────┬────────────┬──────────┬────────────┬──────────────┐")
    print("  │ 迭代次数 │     x_k    │   f(x_k)   │  f'(x_k)   │   λ_k    │   x_{k+1}  │ |Δx|         │")
    print("  ├──────────┼────────────┼────────────┼────────────┼──────────┼────────────┼──────────────┤")
    for it in iter_data:
        print(f"  │ {it[0]:8d} │ {it[1]:10.6f} │ {it[2]:10.6f} │ {it[3]:10.6f} │ {it[4]:8.4f} │ {it[5]:10.6f} │ {it[6]:12.6f} │")
    print("  └──────────┴────────────┴────────────┴────────────┴──────────┴────────────┴──────────────┘\n")

    # ------------------- 打印最终结果 -------------------
    print("【最终结果】")
    print(f"  近似根：x ≈ {xk:.10f}")
    print(f"  |f(x)| = {abs(f(xk)):.3e}")
    print(f"  共迭代 {len(iter_data)} 次（tol={tol}）\n")
    return xk


# ------------------- 示例 1 -------------------
def f1(x):  return x**2 + 10*math.cos(x)
def df1(x): return 2*x - 10*math.sin(x)

newton_downhill(
    f1, df1,
    "f(x) = x^2 + 10*cos(x)",
    "f'(x) = 2x - 10*sin(x)",
    x0       = 1.6,
    tol      = 1e-7,
    max_iter = 20
)

# ------------------- 示例 2 -------------------
def f2(x):  return 1 + math.atan(x) - x
def df2(x): return 1/(1 + x**2) - 1

newton_downhill(
    f2, df2,
    "f(x) = 1 + arctan(x) - x",
    "f'(x) = 1/(1+x^2) - 1",
    x0       = 2.0,
    tol      = 1e-7,
    max_iter = 20
)

# ------------------- 示例 3 -------------------
def f3(x):  return x**2 - 30
def df3(x): return 2*x

newton_downhill(
    f3, df3,
    "f(x) = x^2 - 30",
    "f'(x) = 2x",
    x0       = 5.0,
    tol      = 1e-10,
    max_iter = 20
)
