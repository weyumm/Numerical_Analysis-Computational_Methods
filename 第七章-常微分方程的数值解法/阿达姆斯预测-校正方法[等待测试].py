import numpy as np
import pandas as pd

# 四阶龙格-库塔法（用于生成初始值）
def runge_kutta_4(f, x, y, h):
    """计算单步四阶RK解"""
    k1 = h * f(x, y)
    k2 = h * f(x + h/2, y + k1/2)
    k3 = h * f(x + h/2, y + k2/2)
    k4 = h * f(x + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4)/6

# 阿达姆斯预测-校正方法
def adams_predictor_corrector(f, x, y, h):
    """
    四阶阿达姆斯预测-校正方法
    使用RK4生成前3个初始值后进行预测校正
    """
    y_values = y.copy()
    
    for i in range(3, len(x) - 1):
        # 预测阶段（阿达姆斯-巴什福斯四阶公式）
        f_n = f(x[i], y_values[i])
        f_nm1 = f(x[i-1], y_values[i-1])
        f_nm2 = f(x[i-2], y_values[i-2])
        f_nm3 = f(x[i-3], y_values[i-3])
        
        y_pred = y_values[i] + h/24 * (
            55*f_n - 59*f_nm1 + 37*f_nm2 - 9*f_nm3
        )
        
        # 校正阶段（阿达姆斯-莫尔顿四阶公式）
        y_corr = y_values[i] + h/24 * (
            9*f(x[i+1], y_pred) + 19*f_n - 5*f_nm1 + f_nm2
        )
        
        y_values[i+1] = y_corr
        
    return y_values

# 测试微分方程
def f(x, y):
    """目标微分方程"""
    if abs(y) < 1e-8:  # 防止除以零
        return float('inf')
    return y - 2*x/y

# 主程序
if __name__ == "__main__":
    # 参数设置
    x0, y0 = 0.0, 1.0
    h = 0.1
    x_end = 1.0
    
    # 初始化x值数组
    x_values = np.arange(x0, x_end + h, h)
    n = len(x_values)
    
    # 使用RK4生成前4个初始值
    y_values = np.zeros(n)
    y_values[0] = y0
    for i in range(3):
        y_values[i+1] = runge_kutta_4(f, x_values[i], y_values[i], h)
    
    # 应用阿达姆斯方法
    y_result = adams_predictor_corrector(f, x_values, y_values, h)
    
    # 创建结果表格
    result_df = pd.DataFrame({
        'x': x_values,
        '数值解y': y_result
    })
    
    # 打印格式化结果
    print("阿达姆斯预测-校正方法求解结果")
    print("-" * 40)
    print(result_df.to_string(index=False))
    print("-" * 40)