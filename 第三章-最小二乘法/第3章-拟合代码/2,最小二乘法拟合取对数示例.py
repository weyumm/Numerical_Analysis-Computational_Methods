import numpy as np
import math

class LeastSquaresGenerator:
    def __init__(self, x, y):
        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)
        self.n = len(x)
        self.coefficients = None
        self.predicted = None
        self.errors = None
        
    def fit(self, basis_functions=None, degree=1):
        """
        执行最小二乘拟合
        
        参数:
        basis_functions: 自定义基函数列表 [f0(x), f1(x), ...]
        degree: 多项式次数（当未提供basis_functions时使用）
        """
        # 如果没有提供自定义基函数，则使用多项式基
        if basis_functions is None:
            basis_functions = [lambda x, k=k: x**k for k in range(degree+1)]
        
        # 构造设计矩阵
        X = np.column_stack([func(self.x) for func in basis_functions])
        
        # 构建法方程组
        XTX = X.T @ X
        XTy = X.T @ self.y
        
        # 解方程组
        self.coefficients = np.linalg.solve(XTX, XTy)
        
        # 计算预测值和误差
        self.predicted = X @ self.coefficients
        self.errors = self.y - self.predicted
        
        return {
            'coefficients': self.coefficients,
            'equation': self._format_equation(basis_functions),
            'normal_equations': (XTX, XTy),
            'mse': self._calculate_mse(),
            'max_deviation': np.max(np.abs(self.errors))
        }
    
    def _format_equation(self, basis_functions):
        """格式化输出方程"""
        coeffs = self.coefficients.round(4)
        terms = []
        
        for i, c in enumerate(coeffs):
            if i == 0:
                terms.append(f"{c:.4f}")
            else:
                # 获取基函数名称（如果可用）
                func_name = getattr(basis_functions[i], '__name__', f'f_{i}(x)')
                term = f"{abs(c):.4f}·{func_name}"
                terms.append(f"{'+' if c > 0 else '-'}{term}")
        
        return f"y = {''.join(terms)}"
    
    def _calculate_mse(self):
        """计算均方误差"""
        return np.mean(self.errors**2)
    
    def generate_report(self, basis_functions=None, degree=1):
        """生成完整分析报告"""
        result = self.fit(basis_functions, degree)
        XTX, XTy = result['normal_equations']
        
        report = f"【最小二乘拟合报告】\n"
        report += "\n一、法方程组：\n"
        report += self._format_normal_equations(XTX, XTy)
        report += "\n二、拟合方程：\n"
        report += result['equation'] + "\n"
        report += "\n三、误差分析：\n"
        report += self._format_error_table()
        report += f"\n均方误差(MSE): {result['mse']:.4f}\n"
        report += f"最大偏差: {result['max_deviation']:.4f}"
        
        return report
    
    def _format_normal_equations(self, XTX, XTy):
        """格式化法方程组输出"""
        equations = []
        n = XTX.shape[0]
        
        for i in range(n):
            terms = []
            for j in range(n):
                terms.append(f"{XTX[i,j]:.2f}a_{j}")
            eq = " + ".join(terms) + f" = {XTy[i]:.2f}"
            equations.append(eq)
            
        return "\n".join(equations)
    
    def _format_error_table(self):
        """格式化误差表格"""
        header = "| x值 | 实际值 | 预测值 | 绝对误差 |\n"
        header += "|-----|--------|--------|----------|\n"
        rows = []
        
        for xi, yi, pi in zip(self.x, self.y, self.predicted):
            error = abs(yi - pi)
            rows.append(f"| {xi} | {yi:.2f} | {pi:.2f} | {error:.2f} |")
        return header + "\n".join(rows)

######################## 使用示例 ########################
# 原始数据
t = np.array([1, 2, 4, 8, 16, 32, 64], dtype=float)
W = np.array([4.22, 4.02, 3.85, 3.59, 3.44, 3.02, 2.59], dtype=float)

# 1. 使用自定义基函数进行幂函数拟合
print("="*50 + "\n幂函数拟合（自定义基函数）\n" + "="*50)

# 定义基函数：常数项和ln(t)
basis_funcs = [
    lambda x: np.ones_like(x),  # 常数项
    lambda x: np.sin(x),        # sin(x)
    lambda x: np.cos(x)         # cos(x)
]

"""指数函数拟合
basis_funcs = [
    lambda x: np.ones_like(x),  # 常数项
    lambda x: np.log(x)         # ln(t)
]
对数函数拟合
basis_funcs = [
    lambda x: np.ones_like(x),  # 常数项
    lambda x: x                 # t
]
三角函数拟合
basis_funcs = [
    lambda x: np.ones_like(x),  # 常数项
    lambda x: np.log(x)         # ln(t)
]
"""

# 初始化生成器
lsg = LeastSquaresGenerator(t, W)

# 执行拟合
result = lsg.fit(basis_functions=basis_funcs)

# 提取参数
C = math.exp(result['coefficients'][0])
lambda_ = result['coefficients'][1]

# 计算预测值
W_pred = C * np.exp(lambda_ * np.log(t))  # 等价于 C * t**lambda_

# 输出报告
print(lsg.generate_report(basis_functions=basis_funcs))

# 2. 使用多项式基函数进行二次拟合
print("\n" + "="*50 + "\n二次多项式拟合\n" + "="*50)

# 初始化新生成器
lsg_poly = LeastSquaresGenerator(t, W)

# 执行二次拟合
result_poly = lsg_poly.fit(degree=2)

# 计算预测值
a0, a1, a2 = result_poly['coefficients']
W_pred_poly = a0 + a1*t + a2*t**2

# 输出报告
print(lsg_poly.generate_report(degree=2))

# 3. 比较两种拟合结果
print("\n" + "="*50 + "\n拟合结果比较\n" + "="*50)
print("|  t  | 实际值 | 幂函数预测 | 多项式预测 | 幂函数误差 | 多项式误差 |")
print("|-----|--------|------------|------------|------------|------------|")
for ti, wi, pred, pred_poly in zip(t, W, W_pred, W_pred_poly):
    error = abs(wi - pred)
    error_poly = abs(wi - pred_poly)
    print(f"| {int(ti):2d} | {wi:.2f}  | {pred:.2f}    | {pred_poly:.2f}    | {error:.4f}   | {error_poly:.4f}   |")

# 误差统计
mse = np.mean((W - W_pred)**2)
mse_poly = np.mean((W - W_pred_poly)**2)
max_error = np.max(np.abs(W - W_pred))
max_error_poly = np.max(np.abs(W - W_pred_poly))

print(f"\n幂函数拟合均方误差(MSE): {mse:.4f}")
print(f"多项式拟合均方误差(MSE): {mse_poly:.4f}")
print(f"幂函数拟合最大绝对偏差: {max_error:.4f}")
print(f"多项式拟合最大绝对偏差: {max_error_poly:.4f}")