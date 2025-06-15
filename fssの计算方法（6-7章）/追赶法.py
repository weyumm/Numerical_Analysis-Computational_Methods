from fractions import Fraction
from typing import Union, List

class TriDiagonalSolver:
    """三对角矩阵方程组求解器（追赶法）"""
    
    @staticmethod
    def solve(
        a: List[Union[float, int]], 
        b: List[Union[float, int]], 
        c: List[Union[float, int]], 
        d: List[Union[float, int]],
        verbose: bool = False,
        output_format: str = 'fraction'  # 'fraction' 或 'float'
    ) -> List[Union[Fraction, float]]:
        """
        解三对角矩阵方程组
        
        Args:
            a: 主对角线元素列表（长度n）
            b: 下对角线元素列表（长度n-1）
            c: 上对角线元素列表（长度n-1）
            d: 右端向量（长度n）
            verbose: 是否显示详细步骤
            output_format: 输出格式（'fraction'返回分数，'float'返回浮点数）
            
        Returns:
            解向量（分数或浮点数列表）
            
        Raises:
            ValueError: 输入参数维度不匹配或主对角线元素为零
        """
        n = len(a)
        if len(b) != n-1 or len(c) != n-1 or len(d) != n:
            raise ValueError("维度不匹配：a长度应为n，b/c应为n-1，d应为n")
        
        # 转换为分数矩阵
        matrix = []
        for i in range(n):
            row = [Fraction(0)]*(n+1)
            row[i] = Fraction(a[i])
            if i < n-1: 
                row[i+1] = Fraction(b[i])
            if i > 0:
                row[i-1] = Fraction(c[i-1])
            row[-1] = Fraction(d[i])
            matrix.append(row)
        
        if verbose:
            print("初始矩阵:")
            TriDiagonalSolver._print_matrix(matrix)
        
        # 前向消元
        for i in range(n-1):
            # 检查主元素是否为零
            if matrix[i][i] == 0:
                raise ValueError(f"主对角线元素在第{i+1}行出现零")
            
            # 计算消元因子
            factor = matrix[i+1][i] / matrix[i][i]
            factor_str = str(Fraction(factor).limit_denominator())
            
            # 执行行变换
            for j in range(i, n+1):
                matrix[i+1][j] -= factor * matrix[i][j]
            
            if verbose:
                print(f"\n第{i+1}步消元：r{i+2} = r{i+2} - ({factor_str})*r{i+1}")
                TriDiagonalSolver._print_matrix(matrix)
        
        # 回代求解
        x = [Fraction(0)]*n
        x[-1] = matrix[-1][-1] / matrix[-1][-2]
        
        for i in range(n-2, -1, -1):
            sum_val = Fraction(0)
            for j in range(i+1, n):
                sum_val += matrix[i][j] * x[j]
            x[i] = (matrix[i][-1] - sum_val) / matrix[i][i]
        
        # 格式转换
        if output_format == 'float':
            x = [float(num.limit_denominator()) for num in x]
        elif output_format != 'fraction':
            raise ValueError("output_format必须是'fraction'或'float'")
        
        if verbose:
            print("\n最终解向量:")
            for idx, val in enumerate(x, 1):
                print(f"x[{idx}] = {val}")
        
        return x
    
    @staticmethod
    def _print_matrix(matrix: List[List[Fraction]]):
        """辅助打印方法（分数友好格式）"""
        n = len(matrix)
        col_widths = [max(len(str(row[i]).replace('/', '⁄')) for row in matrix) 
                     for i in range(n+1)]
        
        for row in matrix:
            coeffs = '  '.join(
                str(Fraction(elem).limit_denominator()).replace('/', '⁄').rjust(col_widths[i])
                for i, elem in enumerate(row[:-1])
            )
            rhs = str(Fraction(row[-1]).limit_denominator()).replace('/', '⁄').rjust(col_widths[-1])
            print(f"[{coeffs} | {rhs}]")
        print("-" * (sum(col_widths) + n*3 + 5))

# 使用示例
if __name__ == "__main__":
    # 默认参数示例（与原始问题一致）
    a = [4, 4, 4, 4, 4]
    b = [-1, -1, -1, -1]
    c = [-1, -1, -1, -1]
    d = [100, 200, 200, 200, 100]
    
    # 调用求解器（显示步骤，返回分数）
    solution = TriDiagonalSolver.solve(a, b, c, d, verbose=True, output_format='fraction')
    print("\n解向量（分数形式）：")
    print(solution)
    
    # 调用求解器（返回浮点数）
    float_solution = TriDiagonalSolver.solve(a, b, c, d, output_format='float')
    print("\n解向量（浮点形式）：")
    print(float_solution)