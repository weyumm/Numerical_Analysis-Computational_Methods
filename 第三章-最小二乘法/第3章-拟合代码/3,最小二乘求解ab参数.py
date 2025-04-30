import numpy as np

class LeastSquaresFitter:
    def __init__(self, x, y):
        """
        初始化拟合器
        :param x: 输入数据x值，支持array_like
        :param y: 输入数据y值，支持array_like
        """
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.design_matrix = None
        self.coefficients = None
        self.predicted = None
        self.errors = None
        
    def fit(self, basis_functions):
        """
        执行最小二乘拟合
        :param basis_functions: 基函数列表，例如 [lambda x: 1, lambda x: x**2]
        """
        # 构建设计矩阵
        self.design_matrix = np.column_stack([f(self.x) for f in basis_functions])
        
        # 构建法方程组
        XTX = self.design_matrix.T @ self.design_matrix
        XTy = self.design_matrix.T @ self.y
        
        # 求解参数
        self.coefficients = np.linalg.solve(XTX, XTy)
        
        # 计算预测值和误差
        self.predicted = self.design_matrix @ self.coefficients
        self.errors = self.y - self.predicted
        
        return self
    
    def get_equation(self, coefficient_names=None):
        """
        获取拟合方程字符串
        :param coefficient_names: 系数名称列表，默认使用a0, a1...
        """
        if coefficient_names is None:
            coefficient_names = [f'a{i}' for i in range(len(self.coefficients))]
            
        terms = []
        for coef, name in zip(self.coefficients, coefficient_names):
            terms.append(f"{coef:.4f}{name}")
        return "y = " + " + ".join(terms)
    
    def get_error_analysis(self):
        """
        获取误差分析结果
        :return: 字典包含误差统计量
        """
        return {
            'mse': np.mean(self.errors**2),
            'max_abs_error': np.max(np.abs(self.errors)),
            'r_squared': 1 - np.sum(self.errors**2)/np.sum((self.y - np.mean(self.y))**2)
        }
    
    def generate_report(self, decimal=4):
        """
        生成完整分析报告
        :param decimal: 小数位数
        """
        # 格式化数值显示
        fmt = lambda x: f"{x:.{decimal}f}".rstrip('0').rstrip('.') if '.' in f"{x:.{decimal}f}" else f"{x:.{decimal}f}"
        
        # 法方程组部分
        XTX = self.design_matrix.T @ self.design_matrix
        XTy = self.design_matrix.T @ self.y
        
        report = [
            "="*50,
            "最小二乘拟合分析报告",
            "="*50,
            "\n[法方程组]"
        ]
        
        # 生成法方程表达式
        for i in range(XTX.shape[0]):
            terms = [f"{XTX[i,j]:>{decimal+5}.{decimal}f} a{j}" for j in range(XTX.shape[1])]
            report.append(" + ".join(terms) + f" = {fmt(XTy[i])}")
        
        # 拟合公式
        report += [
            "\n[拟合公式]",
            self.get_equation(),
            "\n[误差分析]",
            f"| {'x':^8} | {'实际值':^8} | {'预测值':^8} | {'绝对误差':^8} |",
            "|"+("-"*9+"|")*4
        ]
        
        # 数据行
        for xi, yi, yp in zip(self.x, self.y, self.predicted):
            report.append(f"| {xi:8.4f} | {yi:8.4f} | {yp:8.4f} | {abs(yi-yp):8.4f} |")
        
        # 统计信息
        stats = self.get_error_analysis()
        report += [
            "\n[统计指标]",
            f"均方误差(MSE): {fmt(stats['mse'])}",
            f"最大绝对误差: {fmt(stats['max_abs_error'])}",
            f"确定系数(R²): {fmt(stats['r_squared'])}",
            "="*50
        ]
        
        return "\n".join(report)

# 使用示例 --------------------------------------------------
if __name__ == "__main__":
    # 原始数据
    x = [19, 25, 31, 38, 44]
    y = [19.0, 32.3, 49.0, 73.3, 97.8]
    # 初始化拟合器
    fitter = LeastSquaresFitter(x, y)
    # 定义拟合模型：y = a + b*x²
    basis_funcs = [
        lambda x: np.ones_like(x),  # 常数项
        lambda x: x**2              # x²项
    ]
    # 执行拟合
    fitter.fit(basis_funcs)
    # 生成报告
    print(fitter.generate_report(decimal=4))
# 使用示例 --------------------------------------------------
# if __name__ == "__main__":
# 示例1：线性拟合 (y = a + bx)​
# 数据：牛顿第二定律实验 (F = ma)
#m = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
#a = np.array([2.0, 1.02, 0.68, 0.49, 0.40])
#fitter = LeastSquaresFitter(m, a)
#basis = [
#    lambda x: np.ones_like(x),  # 常数项
#    lambda x: 1/x              # 1/m 项 (对应F=ma -> a=F/m)
#]
#fitter.fit(basis)
#print(fitter.generate_report(decimal=3))
# 输出参数对应关系：a = F*(1/m) → 斜率b即为力F

#示例2：三次多项式拟合 (y = a + bx + cx² + dx³)​
# 数据：弹簧振动位移-时间关系
#t = np.linspace(0, 2*np.pi, 10)
#y = 2.3*np.sin(t) + 0.5*np.random.randn(10)
#fitter = LeastSquaresFitter(t, y)
#basis = [
#    lambda x: x**0,  # 常数项
#    lambda x: x**1,  # 一次项
#    lambda x: x**2,   # 二次项
#    lambda x: x**3    # 三次项
#]
#fitter.fit(basis)
#print(fitter.generate_report())

#示例5：混合基函数拟合 (y = a + b√x + c·sinx)​
# 数据：非线性系统响应
#x = np.linspace(0, 4*np.pi, 20)
#y = 2.5 + 1.2*np.sqrt(x) + 0.8*np.sin(x) + 0.3*np.random.randn(20)
#fitter = LeastSquaresFitter(x, y)
#basis = [
#    lambda x: np.ones_like(x),    # 常数项
#    lambda x: np.sqrt(x),         # 平方根项
#    lambda x: np.sin(x)            # 正弦项
#]
#fitter.fit(basis)
#print(fitter.generate_report())