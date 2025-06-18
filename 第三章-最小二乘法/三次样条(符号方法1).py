import sympy as sp

# 定义未知系数
a1,b1,c1,d1 = sp.symbols('a1 b1 c1 d1', real=True)
a2,b2,c2,d2 = sp.symbols('a2 b2 c2 d2', real=True)
a3,b3,c3,d3 = sp.symbols('a3 b3 c3 d3', real=True)
#a4,b4,c4,d4 = sp.symbols('a4 b4 c4 d4', real=True)
#a5,b5,c5,d5 = sp.symbols('a5 b5 c5 d5', real=True)

# 定义符号变量 x
x = sp.symbols('x', real=True)

# 定义各区间的三次多项式（局部变量取 (x-区间起点)）
S1 = a1 + b1*x + c1*x**2 + d1*x**3
S2 = a2 + b2*(x-1) + c2*(x-1)**2 + d2*(x-1)**3
S3 = a3 + b3*(x-2) + c3*(x-2)**2 + d3*(x-2)**3

##########################
# 情况 (1): 边界条件 S''(0)=1, S''(3)=0
##########################

# 构造方程组
eqs1 = []

# 插值条件：S(0)=0, S(1)=0, S(2)=0, S(3)=0
eqs1.append(sp.Eq(S1.subs(x,0), 0))      # S1(0)=a1=0
eqs1.append(sp.Eq(S1.subs(x,1), 0))      # S1(1)= a1+b1+c1+d1=0
eqs1.append(sp.Eq(S2.subs(x,1), 0))      # S2(1)=a2=0
eqs1.append(sp.Eq(S2.subs(x,2), 0))      # S2(2)=a2+b2+c2+d2=0
eqs1.append(sp.Eq(S3.subs(x,2), 0))      # S3(2)=a3=0
eqs1.append(sp.Eq(S3.subs(x,3), 0))      # S3(3)=a3+b3+c3+d3=0

# 内部连续性：在 x=1,2 的一阶、二阶连续性
# x=1 处：S1'(1)=S2'(1)
S1p = sp.diff(S1, x)
S2p = sp.diff(S2, x)
S3p = sp.diff(S3, x)
eqs1.append(sp.Eq(S1p.subs(x,1), S2p.subs(x,1)))

# x=1 处：S1''(1)=S2''(1)
S1pp = sp.diff(S1, x, 2)
S2pp = sp.diff(S2, x, 2)
eqs1.append(sp.Eq(S1pp.subs(x,1), S2pp.subs(x,1)))

# x=2 处：S2'(2)=S3'(2)
eqs1.append(sp.Eq(S2p.subs(x,2), S3p.subs(x,2)))

# x=2 处：S2''(2)=S3''(2)
S3pp = sp.diff(S3, x, 2)
eqs1.append(sp.Eq(S2pp.subs(x,2), S3pp.subs(x,2)))

# 边界条件
eqs1.append(sp.Eq(S1pp.subs(x,0), 1))   # S1''(0)= 2*c1 =1  => c1 = 1/2
eqs1.append(sp.Eq(S3pp.subs(x,3), 0))   # S3''(3)= 2*c3+6*d3 =0

# 求解方程组
sol1 = sp.solve(eqs1, [a1,b1,c1,d1,a2,b2,c2,d2,a3,b3,c3,d3], dict=True)
print("情形 (1) 的解：")
sp.pprint(sol1[0])
print("\n")
    
##########################
# 情况 (2): 边界条件 S'(0)=1, S'(3)=0
##########################

# 重新定义未知系数，防止前面的赋值影响
a1,b1,c1,d1 = sp.symbols('a1 b1 c1 d1', real=True)
a2,b2,c2,d2 = sp.symbols('a2 b2 c2 d2', real=True)
a3,b3,c3,d3 = sp.symbols('a3 b3 c3 d3', real=True)

# 重新定义各段多项式
S1 = a1 + b1*x + c1*x**2 + d1*x**3
S2 = a2 + b2*(x-1) + c2*(x-1)**2 + d2*(x-1)**3
S3 = a3 + b3*(x-2) + c3*(x-2)**2 + d3*(x-2)**3

eqs2 = []

# 插值条件：S(0)=0, S(1)=0, S(2)=0, S(3)=0
eqs2.append(sp.Eq(S1.subs(x,0), 0))    # a1=0
eqs2.append(sp.Eq(S1.subs(x,1), 0))    # a1+b1+c1+d1=0
eqs2.append(sp.Eq(S2.subs(x,1), 0))    # a2=0
eqs2.append(sp.Eq(S2.subs(x,2), 0))    # a2+b2+c2+d2=0
eqs2.append(sp.Eq(S3.subs(x,2), 0))    # a3=0
eqs2.append(sp.Eq(S3.subs(x,3), 0))    # a3+b3+c3+d3=0

# 内部连续性条件
S1p = sp.diff(S1, x)
S2p = sp.diff(S2, x)
S3p = sp.diff(S3, x)
eqs2.append(sp.Eq(S1p.subs(x,1), S2p.subs(x,1)))  # S1'(1)=S2'(1)
S1pp = sp.diff(S1, x, 2)
S2pp = sp.diff(S2, x, 2)
eqs2.append(sp.Eq(S1pp.subs(x,1), S2pp.subs(x,1)))  # S1''(1)=S2''(1)
eqs2.append(sp.Eq(S2p.subs(x,2), S3p.subs(x,2)))    # S2'(2)=S3'(2)
S3pp = sp.diff(S3, x, 2)
eqs2.append(sp.Eq(S2pp.subs(x,2), S3pp.subs(x,2)))    # S2''(2)=S3''(2)

# 边界条件：S'(0)=1, S'(3)=0
eqs2.append(sp.Eq(S1p.subs(x,0), 1))  # S1'(0)=b1 = 1
# S3'(3)=? 先利用 S3(3)=a3+b3+c3+d3=0 求出 d3 = -b3-c3，然后 S3'(3)= b3+2c3+3d3 = b3+2c3 - 3(b3+c3)= -2b3 - c3
eqs2.append(sp.Eq(S3p.subs(x,3), 0))

# 求解方程组
sol2 = sp.solve(eqs2, [a1,b1,c1,d1,a2,b2,c2,d2,a3,b3,c3,d3], dict=True)
print("情形 (2) 的解：")
sp.pprint(sol2[0])