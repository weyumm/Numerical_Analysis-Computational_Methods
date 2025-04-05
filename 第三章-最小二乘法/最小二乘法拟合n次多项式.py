import numpy as np
import matplotlib.pyplot as plt

class PolynomialFit:
    def __init__(self, x, y, degree):
        """
        最小二乘多项式拟合类
        :param x: 自变量数据（列表或numpy数组）
        :param y: 因变量数据（列表或numpy数组）
        :param degree: 多项式次数
        """
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.degree = degree
        
        # 异常处理
        if len(self.x) != len(self.y):
            raise ValueError("x和y的长度必须一致")
        if len(self.x) < 2:
            raise ValueError("至少需要2个数据点")
        if self.degree < 0:
            raise ValueError("多项式次数不能为负数")
        if self.degree >= len(self.x):
            raise ValueError("多项式次数不能超过数据点数量-1")
        
        self.coefficients = None
        self.y_pred = None
        self.mse = None
        self.max_deviation = None

    def fit(self):
        """执行最小二乘拟合"""
        self.coefficients = np.polyfit(self.x, self.y, self.degree)
        self.y_pred = np.polyval(self.coefficients, self.x)
        self._calculate_errors()
        return self

    def _calculate_errors(self):
        """计算拟合误差"""
        residuals = self.y - self.y_pred
        self.mse = np.mean(residuals**2)
        self.max_deviation = np.max(np.abs(residuals))

    def get_equation(self):
        """获取多项式方程表达式"""
        terms = []
        for i, coeff in enumerate(self.coefficients[::-1]):
            if i == 0:
                terms.append(f"{coeff:.4f}")
            else:
                terms.append(f"{coeff:+.4f}x{'^{}'.format(i) if i > 1 else ''}")
        return "y = " + " ".join(reversed(terms))

    def plot(self, show=True):
        """绘制拟合曲线"""
        x_fine = np.linspace(min(self.x)-1, max(self.x)+1, 100)
        y_fine = np.polyval(self.coefficients, x_fine)
        
        plt.figure(figsize=(10, 5))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.scatter(self.x, self.y, color='red', label='实验数据')
        plt.plot(x_fine, y_fine, label=f'{self.degree}次拟合曲线')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('最小二乘多项式拟合')
        plt.grid(True)
        
        if show:
            plt.show()

    def __repr__(self):
        return (f"PolynomialFit(degree={self.degree}, "
                f"mse={self.mse:.4f}, max_deviation={self.max_deviation:.4f})")

# ==== 使用示例 ====
if __name__ == "__main__":
    # 实验数据
    x = [2, 4, 6, 8]
    y = [2, 11, 28, 40]

    # 一次拟合
    fit_linear = PolynomialFit(x, y, degree=1).fit()
    print("一次拟合结果：")
    print(fit_linear.get_equation())
    print(f"MSE: {fit_linear.mse:.3f}")
    print(f"最大偏差: {fit_linear.max_deviation:.1f}")
    fit_linear.plot()

    # 二次拟合
    fit_quad = PolynomialFit(x, y, degree=2).fit()
    print("\n二次拟合结果：")
    print(fit_quad.get_equation())
    print(f"MSE: {fit_quad.mse:.4f}")
    print(f"最大偏差: {fit_quad.max_deviation:.2f}")
    fit_quad.plot()

    # 三次拟合
    fit_cubic = PolynomialFit(x, y, degree=3).fit()
    print("\n三次拟合结果：")
    print(fit_cubic.get_equation())
    print(f"MSE: {fit_cubic.mse:.5f}")
    print(f"最大偏差: {fit_cubic.max_deviation:.3f}")
    fit_cubic.plot()