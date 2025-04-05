import numpy as np

class PolynomialFit:
    def __init__(self, x, y, degree):
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.degree = degree
        self.coefficients = None
        self.y_pred = None
        self.mse = None
        self.max_deviation = None

    def fit(self):
        self.coefficients = np.polyfit(self.x, self.y, self.degree)
        self.y_pred = np.polyval(self.coefficients, self.x)
        residuals = self.y - self.y_pred
        self.mse = np.mean(residuals**2)
        self.max_deviation = np.max(np.abs(residuals))
        return self

    def _generate_symbolic_matrix(self):
        n = self.degree + 1
        rows = []
        for i in range(n):
            row = []
            for j in range(n):
                power = i + j
                if power == 0:
                    row.append("Σ1")
                else:
                    row.append(f"Σx_i^{power}" if power > 1 else "Σx_i")
            rows.append(row)
        return rows

    def _generate_symbolic_rhs(self):
        rhs = []
        for i in range(self.degree + 1):
            if i == 0:
                rhs.append("Σy_i")
            else:
                rhs.append(f"Σx_i^{i}y_i" if i > 1 else "Σx_i y_i")
        return rhs

    def generate_report(self):
        report = []
        degree = self.degree
        n = len(self.x)
        
        # 生成符号矩阵
        symbolic_matrix = self._generate_symbolic_matrix()
        symbolic_rhs = self._generate_symbolic_rhs()
        
        # 生成数值矩阵
        if degree == 1:
            A = np.vstack([
                [n, np.sum(self.x)],
                [np.sum(self.x), np.sum(self.x**2)]
            ])
            b = [np.sum(self.y), np.sum(self.x*self.y)]
        else:
            A = np.vstack([
                [n, np.sum(self.x), np.sum(self.x**2)],
                [np.sum(self.x), np.sum(self.x**2), np.sum(self.x**3)],
                [np.sum(self.x**2), np.sum(self.x**3), np.sum(self.x**4)]
            ])
            b = [np.sum(self.y), np.sum(self.x*self.y), np.sum(self.x**2*self.y)]
        
        # 转换为文本格式
        numeric_matrix = "\n".join("  ".join(f"{cell:8.0f}" for cell in row) for row in A)
        numeric_rhs = "\n".join(f"{val:8.0f}" for val in b)
        
        # 构建符号矩阵文本
        symbolic_text = []
        for row in symbolic_matrix:
            symbolic_text.append("  ".join(f"{term:^8s}" for term in row))
        symbolic_text = "\n".join(symbolic_text)
        
        symbolic_rhs_text = "\n".join(f"{term:^8s}" for term in symbolic_rhs)
        
        # 生成报告
        report.append(f"{'='*50}")
        report.append(f"拟合次数：{degree}次多项式")
        
        # 符号法方程组
        report.append("\n符号法方程组：")
        report.append("矩阵形式：")
        report.append(f"[\n{symbolic_text}\n]")
        report.append("参数向量：")
        report.append("[a, b]" if degree == 1 else "[a, b, c]")
        report.append("右侧向量：")
        report.append(f"[\n{symbolic_rhs_text}\n]")
        
        # 数值法方程组
        report.append("\n代入数值后的法方程组：")
        report.append(f"矩阵：\n{numeric_matrix}")
        report.append(f"右侧向量：\n{numeric_rhs}")
        
        # 系数解
        coeffs = [f"{c:.4f}" for c in reversed(self.coefficients)]
        coeff_vars = ["a", "b", "c"][:len(coeffs)]
        coeff_str = "  ".join([f"{v} = {c}" for v, c in zip(coeff_vars, coeffs)])
        report.append("\n解得系数：")
        report.append(coeff_str)
        
        # 生成表达式
        terms = []
        for i, c in enumerate(reversed(self.coefficients)):
            if i == 0:
                terms.append(f"{c:.4f}")
            else:
                terms.append(f"{'+' if c > 0 else ''}{c:.4f}x{'²' if i==2 else '' if i==1 else '³'}")
        equation = "y = " + "  ".join(terms)
        report.append(f"\n拟合多项式：\n{equation}")
        
        # 预测值与误差
        report.append("\n预测值与误差：")
        header = "|".join(f"{s:^10}" for s in ["x", "y实际", "y预测", "误差"])
        report.append("-" * len(header))
        report.append(header)
        report.append("-" * len(header))
        for xi, yi, yp in zip(self.x, self.y, self.y_pred):
            error = yp - yi
            line = f"{xi:10.0f}|{yi:10.0f}|{yp:10.2f}|{error:10.2f}"
            report.append(line)
        
        # 误差指标
        report.append(f"\n均方误差：{self.mse:.4f}")
        report.append(f"最大偏差：{self.max_deviation:.4f}")
        report.append(f"{'='*50}")
        
        return "\n".join(report)

# 示例用法
if __name__ == "__main__":
    x = [2, 4, 6, 8]
    y = [2, 11, 28, 40]
    
    print("一次多项式拟合报告：")
    fit1 = PolynomialFit(x, y, 1).fit()
    print(fit1.generate_report())
    
    print("\n二次多项式拟合报告：")
    fit2 = PolynomialFit(x, y, 2).fit()
    print(fit2.generate_report())

    print("\n三次多项式拟合报告：")
    fit3 = PolynomialFit(x, y, 3).fit()
    print(fit3.generate_report())