import numpy as np
from typing import Union, List, Optional

class TriDiagonalSolver:
    """三对角矩阵方程组数值求解器（带矩阵可视化）"""
    
    @staticmethod
    def solve(
        a: List[Union[float, int]], 
        b: List[Union[float, int]], 
        c: List[Union[float, int]], 
        d: List[Union[float, int]],
        verbose: bool = False,
        tol: float = 1e-10  # 主元素零判定阈值
    ) -> np.ndarray:
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        c = np.array(c, dtype=float)
        d = np.array(d, dtype=float)
        n = len(a)
        
        # 输入验证
        if len(b) != n-1 or len(c) != n-1 or len(d) != n:
            raise ValueError("维度不匹配：a长度应为n，b/c应为n-1，d应为n")
        
        # 初始化工作向量
        alpha = np.empty(n)
        beta = np.empty(n-1)
        d_prime = np.empty(n)
        
        # 创建增广矩阵副本用于可视化
        if verbose:
            augmented = np.zeros((n, n+1))
            for i in range(n):
                augmented[i, i] = a[i]
                if i > 0:
                    augmented[i, i-1] = c[i-1]
                if i < n-1:
                    augmented[i, i+1] = b[i]
                augmented[i, -1] = d[i]
            TriDiagonalSolver._print_matrix(augmented, title="初始矩阵")
        
        # 追过程（LU分解）
        alpha[0] = a[0]
        d_prime[0] = d[0]
        
        for i in range(1, n):
            # 计算当前步的beta和alpha
            beta[i-1] = c[i-1] / alpha[i-1]
            alpha[i] = a[i] - beta[i-1] * b[i-1]
            d_prime[i] = d[i] - beta[i-1] * d_prime[i-1]
            
            # 主元素检查
            if abs(alpha[i]) < tol:
                raise ValueError(f"主元素在第{i}行接近零：{alpha[i]}")
            
            # 可视化当前步骤
            if verbose:
                current_augmented = np.copy(augmented)
                # 更新当前行的系数
                current_augmented[i, i-1] = 0.0
                current_augmented[i, i] = alpha[i]
                current_augmented[i, -1] = d_prime[i]
                # 后续行的上对角线元素（如果需要显示未变化的部分）
                # 注意：实际计算中这些元素已被beta隐式处理
                TriDiagonalSolver._print_matrix(current_augmented, 
                    title=f"第{i}步消元（r{i+1} = r{i+1} - {beta[i-1]:.4f}*r{i}）")
        
        # 赶过程（回代）
        x = np.empty(n)
        x[-1] = d_prime[-1] / alpha[-1]
        
        for i in range(n-2, -1, -1):
            x[i] = (d_prime[i] - b[i] * x[i+1]) / alpha[i]
        
        if verbose:
            print("\n最终解向量：")
            for idx in range(n):
                print(f"x[{idx+1}] = {x[idx]:.6f}")
        
        return x
    
    @staticmethod
    def _print_matrix(matrix: np.ndarray, title: str = "", precision: int = 4):
        """格式化打印矩阵"""
        print(f"\n{title}")
        n = matrix.shape[0]
        for i in range(n):
            row = [f"{num:.{precision}f}" for num in matrix[i]]
            print(f"[{'  '.join(row[:-1])} | {row[-1]}]")
        print("-" * (8*(matrix.shape[1]+1)))

# 使用示例
if __name__ == "__main__":
    # 示例参数
    a = [4, 4, 4, 4, 4]
    b = [-1, -1, -1, -1]
    c = [-1, -1, -1, -1]
    d = [100, 200, 200, 200, 100]
    """
    示例参数的矩阵形式如下：
    [ 4, -1,  0,  0,  0 ,| 100]
    [-1,  4, -1,  0,  0,| 200]
    [ 0, -1,  4, -1,  0,| 200]
    [ 0,  0, -1,  4, -1,| 200]
    [ 0,  0,  0, -1,  4,| 100]
    """
    
    # 调用求解器（显示详细步骤）
    solution = TriDiagonalSolver.solve(a, b, c, d, verbose=True)
    
    print("\n最终解向量：")
    print(solution)