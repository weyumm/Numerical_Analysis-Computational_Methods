import math

# 不动点函数 g(x) = e^(-x)
def g(x):
    return math.exp(-x)

# 史蒂芬森迭代法
def steffensen_method(g, x0, tol=1e-6, max_iter=100):
    print(f"{'k':<5} {'xk':<15} {'yk':<15} {'zk':<15}")
    print("-" * 50)
    
    for k in range(max_iter):
        yk = g(x0)
        zk = g(yk)
        
        # 打印当前迭代信息
        print(f"{k:<5} {x0:<15.8f} {yk:<15.8f} {zk:<15.8f}")
        
        # 计算分母
        denominator = zk - 2 * yk + x0
        if abs(denominator) < 1e-12:  # 避免除以零
            print("Denominator too small. Stopping.")
            return None
        
        # 史蒂芬森更新公式
        x_new = x0 - (yk - x0) ** 2 / denominator
        
        # 判断是否满足精度要求
        if abs(x_new - x0) < tol:
            print(f"\nConverged at iteration {k}:")
            print(f"Root ≈ {x_new:.10f}")
            return x_new
        
        x0 = x_new
    
    print("Maximum iterations reached without convergence.")
    return None

# 设置初始值
steffensen_method(g, x0=0.5)