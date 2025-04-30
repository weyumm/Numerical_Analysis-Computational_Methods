import numpy as np
class LeastSquaresGenerator:
    def __init__(self, x, y):
        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)
        self.n = len(x)
        self.coefficients = None
        self.predicted = None
        self.errors = None
        
    def fit(self, degree=1):
        """执行多项式拟合，支持一次(degree=1)和二次(degree=2)"""
        # 构造设计矩阵
        X = np.column_stack([self.x**k for k in range(degree+1)])
        
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
            'equation': self._format_equation(degree),
            'normal_equations': (XTX, XTy),
            'mse': self._calculate_mse(),
            'max_deviation': np.max(np.abs(self.errors))
        }
    
    def _format_equation(self, degree):
        """格式化输出多项式方程"""
        coeffs = self.coefficients.round(4)
        terms = []
        
        for i, c in enumerate(coeffs):
            if i == 0:
                terms.append(f"{c:.4f}")
            else:
                term = f"{abs(c):.4f}x"
                if i > 1:
                    term += f"^{i}"
                terms.append(f"{'+' if c > 0 else '-'}{term}")
        
        return f"y = {''.join(terms)}"
    
    def _calculate_mse(self):
        """计算均方误差"""
        return np.mean(self.errors**2)
    
    def generate_report(self, degree=1):
        """生成完整分析报告"""
        result = self.fit(degree)
        XTX, XTy = result['normal_equations']
        
        report = f"【{degree}次拟合报告】\n"
        report += "\n一、法方程组：\n"
        report += self._format_normal_equations(XTX, XTy)
        report += "\n二、拟合多项式：\n"
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
import math
# 原始数据
t = np.array([1, 2, 4, 8, 16, 32, 64], dtype=float)
W = np.array([4.22, 4.02, 3.85, 3.59, 3.44, 3.02, 2.59], dtype=float)
# 数据线性化处理
X = np.log(t)
Y = np.log(W)
# 初始化生成器
lsg = LeastSquaresGenerator(X, Y)
# 执行一次多项式拟合
report = lsg.fit(degree=1)
# 提取参数并转换
a, lambda_ = report['coefficients']
C = math.exp(a)
# 计算预测值
W_pred = C * t**lambda_

# 生成最终报告
print("="*50 + "\n幂函数拟合结果\n" + "="*50)
# 1. 输出法方程组
print("\n[法方程组]")
for eq in lsg._format_normal_equations(*report['normal_equations']).split('\n'):
    print(eq.replace('a_0', 'lnC').replace('a_1', 'λ'))
# 2. 输出拟合公式
print(f"\n[拟合公式]\nW = {C:.4f}·t^{lambda_:.4f}")
# 3. 误差分析
print("\n[误差分析]")
print("|  t  | 实际值 | 预测值 | 绝对误差 | 相对误差(%) |")
print("|-----|--------|--------|----------|------------|")
for ti, wi, pred in zip(t, W, W_pred):
    abs_error = abs(wi - pred)
    rel_error = abs_error/wi*100
    print(f"| {int(ti):2d} | {wi:.2f}  | {pred:.2f}  | {abs_error:.4f}  | {rel_error:.2f}%    |")

print(f"\n[误差统计]")
print(f"均方误差(MSE): {np.mean((W - W_pred)**2):.4f}")
print(f"最大绝对偏差: {np.max(np.abs(W - W_pred)):.4f}")