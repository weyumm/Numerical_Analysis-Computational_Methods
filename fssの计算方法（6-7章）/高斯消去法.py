import numpy as np
def gaussian_elimination_np(A, b):
    n = A.shape[0]
    # 构造增广矩阵 [A | b]
    augmented = np.hstack((A.astype(float), b.reshape(-1, 1).astype(float)))
    # 前向消元
    for i in range(n):
        # 部分选主元：找到当前列中绝对值最大的行
        max_row = np.argmax(np.abs(augmented[i:, i])) + i
        # 交换行
        augmented[[i, max_row]] = augmented[[max_row, i]]
        # 检查主元是否接近零
        if np.abs(augmented[i, i]) < 1e-9:
            raise ValueError("矩阵为奇异矩阵，无唯一解")
        # 消去下方行的当前列元素
        for j in range(i+1, n):
            factor = augmented[j, i] / augmented[i, i]
            augmented[j, i:] -= factor * augmented[i, i:]
    # 回代求解
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (augmented[i, -1] - augmented[i, i+1:n].dot(x[i+1:])) / augmented[i, i]
    return x
#矩阵赋值函数
def matrix(n,x_range,y_range,y0,yn,tp,mat):
    h=[]
    mu=[]
    lbd=[]
    g=[]
    for j in range(n):
        h.append(x_range[j+1]-x_range[j])
    for k in range(n-1):
        mu.append(h[k]/(h[k]+h[k+1]))
    for l in range(n-1):
        lbd.append(1-mu[l])
    for m in range(n-1):
        g.append(6/(h[m]+h[m+1])*((y_range[m+1]-y_range[m+2])/(x_range[m+1]-x_range[m+2])-(y_range[m]-y_range[m+1])/(x_range[m]-x_range[m+1])))
    #对所求矩阵进行分类，tp表示边界条件类型，mat表示矩阵类型
    if tp==1:
        if mat==1:
            A1=np.zeros((n+1,n+1))
            for i in range(n+1):
                A1[i][i]=2
            A1[0][1]=1
            A1[n][n-1]=1
            for p in range(n-1):
                A1[p+1][p]=mu[p]
            for q in range(n-1):
                A1[q+1][q+2]=lbd[q]
            return A1
        if mat==2:
            b1=np.zeros((n+1,1))
            for r in range(n-1):
                b1[r+1][0]=g[r]
                b1[0][0]=6/h[0]*((y_range[0]-y_range[1])/(x_range[0]-x_range[1])-y0)
                b1[n][0]=6/h[n-1]*(yn-(y_range[n-1]-y_range[n])/(x_range[n-1]-x_range[n]))
            return b1
    if tp==2:
        if mat==1:
            A2=np.zeros((n-1,n-1))
            for i in range(n-1):
                A2[i][i]=2
            for p in range(1,n-1):
                A2[p][p-1]=mu[p]
            for q in range(n-2):
                A2[q][q+1]=lbd[q]
            return A2
        if mat==2:
            b2=np.zeros((n-1,1))
            for r in range(n-2):
                b2[r][0]=g[r]
                b2[0][0]-=mu[0]*y0
                b2[n-2][0]-=lbd[n-2]*yn
            return b2
def cubic_spline(x, y, M):
    n = len(x)
    coeffs = []
    for i in range(n-1):
        h_i = x[i+1] - x[i]
        a = y[i]
        b = (y[i+1] - y[i])/h_i - (2*M[i] + M[i+1])*h_i/6
        c = M[i]/2
        d = (M[i+1] - M[i])/(6*h_i)
        coeffs.append((a, b, c, d))
    return coeffs
#输入矩阵初始值
x_range1=[0,1,2,3]
y_range1=[0,0,0,0]
n1=3
y01=1
yn1=0
y02=1
yn2=0
A_1=matrix(n1,x_range1,y_range1,y01,yn1,1,1)
b_1=matrix(n1,x_range1,y_range1,y01,yn1,1,2)
A_2=matrix(n1,x_range1,y_range1,y02,yn2,2,1)
b_2=matrix(n1,x_range1,y_range1,y02,yn2,2,2)
M1=gaussian_elimination_np(A_1,b_1)
print(M1)
M2=gaussian_elimination_np(A_2,b_2)
M2=np.append(M2,yn2)
M2=np.insert(M2,0,y02)
print(M2)
print(cubic_spline(x_range1,y_range1,M1))
print(cubic_spline(x_range1,y_range1,M2))