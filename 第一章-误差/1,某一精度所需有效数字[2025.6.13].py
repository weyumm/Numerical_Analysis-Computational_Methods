import math

def get_leading_digit(x):
    """获取数值的首位非零数字"""
    if x == 0:
        return 0
    x_abs = abs(x)
    exponent = math.floor(math.log10(x_abs))
    leading_digit = int(x_abs / (10 ** exponent))
    return leading_digit

def calculate_significant_digits(x, epsilon, error_type='relative', multiplier=1):
    """
    计算满足误差要求的最小有效数字位数
    
    参数:
    x - 真实值（可直接输入表达式如math.pi**8）
    epsilon - 误差限（绝对/相对误差）
    error_type - 误差类型 ('absolute'/'relative')
    multiplier - 误差传播系数（如求π⁸时取8）
    
    返回: 最小有效数字位数
    """
    print("\n【初始参数】")
    print(f"输入值: {x:.10f}")
    print(f"目标误差限: {epsilon} ({error_type}, 系数{multiplier})")
    
    # Step 1: 建立数学模型
    print("\n【Step 1】数学建模")
    if error_type == 'relative':
        print(f"根据误差传播公式：dx/x ≤ ε/{multiplier}")
        epsilon = epsilon / multiplier
        print(f"修正后误差限: {epsilon:.6f}")
    
    # Step 2: 确定首位数字
    a1 = get_leading_digit(x)
    print(f"\n【Step 2】首位数字分析")
    print(f"真实值的首位非零数字 a₁ = {a1}")
    
    # Step 3: 迭代计算有效位数
    print(f"\n【Step 3】迭代验证有效位数 (使用公式: 1/(2(a₁+1))·10^-(n-1) ≤ ε)")
    n = 1
    while True:
        # 根据知识库公式修正计算
        error_limit = (1 / (2 * (a1 + 1))) * (10 ** (-(n - 1)))
        print(f"n={n}: 理论误差限={error_limit:.6e} | 目标={epsilon:.6e}", end='')
        
        if error_limit <= epsilon:
            print(" ✅ 满足条件")
            break
        print(" ❌ 不满足")
        n += 1
    
    # Step 4: 结果验证
    print(f"\n【Step 4】结果验证")
    print(f"当n={n}时:")
    print(f"理论误差限: {error_limit:.6e}")
    print(f"实际误差限: 0.5×10^(-{n-1}) = {0.5*10**(-(n-1)):.6e}")
    
    return n

# 示例测试入口
if __name__ == "__main__":
    print("【问题】要求 π⁸ 的相对误差 ≤ 0.01%，π 至少应取几位有效数字？")
    print("\n【求解过程】")
    
    # 支持直接输入表达式
    true_value = math.pi**8  # 直接输入目标值
    epsilon = 0.0001         # 0.01%相对误差
    
    min_digits = calculate_significant_digits(
        true_value, 
        epsilon, 
        error_type='relative',
        multiplier=8  # π⁸的误差传播系数
    )
    
    print(f"\n【结论】π 至少应取 {min_digits} 位有效数字")