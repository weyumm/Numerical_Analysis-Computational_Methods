# 第二章 插值法

## 2.1 插值法概述
在工程和科学实验中，常需从观测数据$(x_i, y_i), i = 0, 1, \cdots, n$ 揭示变量关系，可用数据拟合或插值构建近似函数$y = f(x)$。拟合旨在求整体误差小的近似函数，不保证$y_i = f(x_i)$；插值要求函数在观测点处满足$y_i = f(x_i)$。本章聚焦插值法，常用类型有多项式插值、分段低次插值、有理函数插值和高维插值。

## 2.2 插值多项式及存在唯一性
### （一）定义与原理
设函数$f(x)$在$[a, b]$上有定义，已知$n + 1$个互异节点$x_0, x_1, \cdots, x_n$的函数值$f(x_0), f(x_1), \cdots, f(x_n)$ ，若存在不超过$n$次的多项式$P_n(x)$，满足$P_n(x_i) = f(x_i) (i = 0, 1, \cdots, n)$ ，则$P_n(x)$为$f(x)$的$n$次插值多项式，$x_i$为插值节点，$f(x)$为被插函数，$R_n(x) = f(x) - P_n(x)$为插值余项。
设$P_n(x)=a_0 + a_1x + \cdots + a_{n - 1}x^{n - 1} + a_nx^n$，根据$P_n(x_i) = f(x_i)$可列出线性方程组，写成矩阵形式$Xa = y$，其中$X$是范德蒙德（Vandermonde）矩阵。

### （二）存在唯一性证明
由于节点互异，$X$的行列式$\det(X)\neq0$，根据克拉默（Grammer）法则，线性方程组$Xa = y$有唯一解，即插值多项式$P_n(x)$存在且唯一。不过，通过求解该方程组求插值多项式计算量较大，实际应用不便。

## 2.3 Lagrange插值
### （一）基函数构造
将插值多项式$P_n(x)$写成向量乘积形式，为便于计算系数，选择另一组基函数$l_0(x), l_1(x), \cdots, l_{n - 1}(x), l_n(x)$ 。当满足特定条件时，推导出Lagrange插值基函数$l_i(x)=\prod_{\substack{j = 0\\j\neq i}}^{n}\frac{x - x_j}{x_i - x_j}$，记$\omega_{n + 1}(x)=\prod_{i = 0}^{n}(x - x_i)$ ，则$l_i(x)=\frac{\omega_{n + 1}(x)}{(x - x_i)\omega_{n + 1}'(x)}$ 。

### （二）插值多项式
满足条件的$n$次Lagrange插值多项式为$L_n(x)=\sum_{i = 0}^{n}f(x_i)l_i(x)$ 。

### （三）MATLAB实现
```matlab
function varargout=lagrange_interp(xdata,ydata,xi)
% LAGRANGE_INTERP, Lagrange插值
n=length(xdata); % 求向量xdata的长度
if length(unique(xdata))<n % 若输入的点出现相同时，给出错误提示
    error('输入的点必须是互异的.')
end
L=zeros(n); % 存储插值基函数
for i=1:n
    px=poly(xdata([1:i-1,i+1:n])); % 构造以x_j为根的多项式(j=1:i-1,i+1:n)
    L(i,:)=px/polyval(px,xdata(i)); % 求插值基函数并存储
end
y=sum(bsxfun(@times,L,ydata(:))); % 求插值多项式
if nargin==3 % 若输入参数为3个
    y=polyval(y,xi); % 根据插值多项式求指定点处的值
end
[varargout{1:2}]=deal(y,... % 第1个输出参数为插值多项式或其在某点处的值
    L); % 第2个输出参数为插值基函数
```

### （四）应用实例
- **例6 - 1**：在铁路缓冲区问题中，利用Lagrange多项式插值法，根据观测的铁路缓和曲线上4点坐标，调用`lagrange_interp`函数可求解缓和曲线方程，并通过绘图展示。
- **例6 - 2**：对于函数$f(x) = x^2\cos x$，先生成区间$[2, 5]$上的4个等距离散点，调用`lagrange_interp`函数进行插值，计算$f(3.5)$的值并估计截断误差，通过绘图和计算步骤展示了整个过程。 

### 2.4 Newton插值
#### （一）Lagrange插值的缺陷与Newton插值的引入
Lagrange多项式插值法公式结构紧凑，理论分析方便，但当插值节点增加时，所有插值基函数都要改变，计算需重新开始，缺乏“继承性”。为克服该缺点，引入Newton插值法。

#### （二）差商的概念与计算
1. **定义**：
    - 设函数\(f(x)\)在\([a,b]\)上有\(n + 1\)个互异节点\(x_0,x_1,\cdots,x_n\) 。\(f(x)\)关于节点\(x_i,x_j\)的1阶差商（均值）为\(f[x_i,x_j]=\frac{f(x_j)-f(x_i)}{x_j - x_i}\) 。
    - 1阶差商\(f[x_i,x_j]\)和\(f[x_j,x_k]\)的差商，即\(f(x)\)关于节点\(x_i,x_j\)和\(x_k\)的二阶差商为\(f[x_i,x_j,x_k]=\frac{f[x_j,x_k]-f[x_i,x_j]}{x_k - x_i}\) 。
    - 一般地，\(k - 1\)阶差商的差商为\(k\)阶差商，\(f[x_0,x_1,\cdots,x_{k - 1},x_k]=\frac{f[x_1,x_2,\cdots,x_k]-f[x_0,x_1,\cdots,x_{k - 1}]}{x_k - x_0}\) 。
    - 约定\(f[x_i]=f(x_i)\)为\(f(x)\)关于节点\(x_i\)的零阶差商。
2. **差商与导数的关系**：\(k\)阶差商和\(k\)阶导数有关系\(f[x_0,x_1,\cdots,x_{k - 1},x_k]=\frac{f^{(k)}(\eta)}{k!}\) ，其中\(\eta\in(\min\{x_0,x_1,\cdots,x_{k - 1},x_k\},\max\{x_0,x_1,\cdots,x_{k - 1},x_k\})\) 。
3. **MATLAB实现（构造差商表）**：
```matlab
function d=diffquot(x,y,m)
% DIFFQUOT 差商表
n=length(x); % 数据向量的长度
if nargin==2
    m=n-1; % 默认差商阶数
end
if m>=n % 判断差商阶数
    error('差商阶数应小于数据长度.')
end
if length(unique(x))<n % 若输入的点出现相同时，给出错误提示
    error('输入的点必须是互异的.')
end
if ~isnumeric(y)
    y=feval(y,x); % 若y用函数形式，则求函数y在点x处的值
end
d(:,[1,2])=[x(:),y(:)]; % d的前两列
for k=1:m
    d(1:n-k,k+2)=diff(d(1:n-k+1,k+1))./(d(k+1:n,1)-d(1:n-k,1)); % 计算差商
end
```
4. **示例（例6 - 3）**：对于函数\(y = \ln x\) ，在点\(x_0 = 1,x_1 = 3,x_2 = 4,x_3 = 6\)处，调用`diffquot`函数可构造差商表。

#### （三）Newton插值多项式
1. **形式与系数推导**：
    - Newton插值多项式一般形式为\(N_n(x)=a_0 + a_1(x - x_0)+a_2(x - x_0)(x - x_1)+\cdots+a_n(x - x_0)(x - x_1)\cdots(x - x_{n - 1})\) 。
    - 由插值条件\(N_n(x_i)=f(x_i)\) ，可得\(a_0 = f(x_0)=f[x_0]\) ，\(a_1 = f[x_0,x_1]\) ，一般地\(a_i = f[x_0,x_1,\cdots,x_i]\) ，\(i = 0,1,\cdots,n\) 。从而得到\(n\)次Newton插值多项式\(N_n(x)=f(x_0)+f[x_0,x_1](x - x_0)+f[x_0,x_1,x_2](x - x_0)(x - x_1)+\cdots + f[x_0,x_1,\cdots,x_n](x - x_0)(x - x_1)\cdots(x - x_{n - 1})\) 。
    - 也可从差商定义推导，通过\(f(x)\)各阶差商定义式依次代入，得到\(f(x)=N_n(x)+R_n(x)\) ，同时得出插值多项式和插值余项。
2. **MATLAB实现**：
```matlab
function varargout=newton_interp(xdata,ydata,xi)
% NEWTON_INTERP Newton插值法
n=length(xdata); % 求向量xdata的长度
if length(unique(xdata))<n % 若输入的点出现相同时，给出错误提示
    error('输入的点必须是互异的.')
end
d=[xdata(:),ydata(:),zeros(n,n-1)]; % 存储差商表
N=zeros(n); % 存储每一个基函数
N(1,end)=1; % 第一个基函数为1
for k=1:n-1
    d(1:n-k,k+2)=diff(d(1:n-k+1,k+1))./(d(k+1:n,1)-d(1:n-k,1)); % 计算差商
    N(k+1,n-k:n)=poly(xdata(1:k)); % 存储基函数
end
y=sum(bsxfun(@times,N,d(:,2:n+1).')); % 求牛顿插值多项式
if nargin==3
    y=polyval(y,xi); % 根据牛顿插值多项式求指定点的值
end
[varargout{1:2}]=deal(y,... % 第1个输出参数为插值多项式或其在某点处的值
    N); % 第2个输出参数为插值基函数
```
3. **示例（例6 - 4）**：利用Newton插值法求解铁路缓和曲线方程问题（同例6 - 1数据），通过在命令窗口执行相关语句，可得到插值多项式并绘制曲线 。 

## 2.5 三次样条插值
#### （一）引入背景
高阶多项式插值存在计算复杂且可能出现Runge现象的问题，分段插值虽然计算简单且具有一致收敛性，但光滑性较差。而在一些实际问题中，如飞机机翼外形设计、内燃机进排气门的凸轮曲线等，对插值函数的光滑性要求较高，不仅需要连续，还要求有连续的曲率。早期工程师绘图时用富有弹性的细木条（样条）固定在样值点上自由弯曲得到样条曲线，在此基础上发展出了三次样条插值。

#### （二）三次样条函数及插值函数定义
给定区间$[a, b]$上的$n + 1$个节点$a = x_0 < x_1 < \cdots < x_n = b$及函数值$f(x_i) (i = 0, 1, \cdots, n)$ ，若函数$S(x)$满足：
- 在每一个小区间$[x_i, x_{i + 1}]$上是三次多项式。
- 在区间$[a, b]$上存在连续的二阶导数。
则称$S(x)$是区间$[a, b]$上的三次样条函数。若$S(x)$还满足$S(x_i) = f(x_i) (i = 0, 1, \cdots, n)$ ，则称$S(x)$是区间$[a, b]$上的三次样条插值函数。

$S(x)$在每个子区间$[x_i, x_{i + 1}]$上的表达式为$S(x)=A_i + B_ix + C_ix^2 + D_ix^3$ ，$i = 0, 1, \cdots, n - 1$ ，其中$A_i, B_i, C_i, D_i$是待定系数，需满足插值条件$S(x_i) = f(x_i)$和连接条件$\begin{cases}S(x_i + 0)=S(x_i - 0)=f(x_i)\\S'(x_i + 0)=S'(x_i - 0)\\S''(x_i + 0)=S''(x_i - 0)\end{cases}$ ，$i = 1, 2, \cdots, n - 1$ 。这些条件共给出$n + 1 + 3(n - 1)=4n - 2$个方程。

#### （三）边界条件
为保证参数的唯一性，还需两个方程，即边界条件。常用的边界条件有以下3种类型：
- **m边界条件**：$S'(x_0)=m_0$，$S'(x_n)=m_n$，$m_0, m_n$一般是端点处的一阶导数值。若$m_0, m_n$不确定，可通过Lagrange插值得到一阶导数的方程来确定。
- **M边界条件**：$S''(x_0)=M_0$，$S''(x_n)=M_n$，$M_0, M_n$一般是端点处的二阶导数值。当$S''(x_0)=S''(x_n)=0$时，称为自然边界条件，此时的$S(x)$称为自然样条函数。
 - **周期性边界条件**：当函数$f(x)$是周期为$b - a$的周期函数时，$S(x)$也是周期为$b - a$的周期函数，即$S'(x_0)=S'(x_n)$，$S''(x_0)=S''(x_n)$ 。

任意一种边界条件都能补充两个方程，加上原来的$4n - 2$个方程，就可以唯一确定$4n$个系数。但用待定系数法求解，当$n$较大时计算量很大。

#### （四）MATLAB实现
```matlab
function varargout=pcsinterp(xdata,ydata,varargin)
% PCSINTERP 分段三次样条插值
n=length(xdata); % 求向量xdata的长度
if length(unique(xdata))<n % 若输入的点出现相同时，给出错误提示
    error('输入的点必须是互异的.')
end
if nargin==2
    args={'second'}; % 默认是自然边界条件，即首末节点的二阶导数值均为0
else
    args=varargin; % 将输入参数元胞数组赋给变量args
end
if isnumeric(args{1})
    xi=args{1}; % 插值点
    args=args(2:end); % 边界条件及边界条件值
end
conds=args{1}; % 边界条件
if numel(args)==2
    valconds=args{2}; % args的第二个元素即为边界条件值
end
hx=diff(xdata); % 求各个区间的长度
hy=diff(ydata); % 求节点处函数值的差分
mu=reshape(hx(1:n-2)./(hx(1:n-2)+hx(2:n-1)),[],1); % 系数向量
la=1-mu; % 系数向量
d=6*reshape(diff(hy./hx)./(hx(1:n-2)+hx(2:n-1)),[],1); % 右端向量
pp=zeros(n-1,6); % 存储每个分段三次样条插值函数以及对应的分段区间
y=@(x)zeros(size(x)); % 分段三次样条插值函数累加量
if strcmpi(conds(1:3),'fir') % m边界条件
    if numel(args)==1 % 若边界条件值默认，则用Lagrange插值求得
        y0=lagrange_interp(xdata(1:4),ydata(1:4));
        yn=lagrange_interp(xdata(n-3:n),ydata(n-3:n));
        valconds=[polyval(polyder(y0),xdata(1)),...
            polyval(polyder(yn),xdata(n))]; % Lagrange插值求m0和m_n的值
    end
    M=chase([mu;1],2*ones(n,1),[1;la],...
        [6*(hy(1)/hx(1)-valconds(1)/hx(1));d;
        6*(valconds(2)-hy(n-1)/hx(n-1))/hx(n-1)]); % 追赶法求Mi (i=1:n)
elseif strcmpi(conds(1:3),'sec') % M边界条件
    if numel(args)==1
        valconds=[0,0]; % 若边界条件默认，则设为自然边界条件
    end
    M1=chase(mu(2:n-2),2*ones(n-2,1),la(1:n-3),...
        % 追赶法求Mi (i=2:n-1)
        [d(1)-mu(1)*valconds(1);d(2:n-3);d(n-2)-la(n-2)*valconds(2)]);
    M=[valconds(1);M1;valconds(2)]; % 所有节点处的二阶导数值
elseif strcmpi(conds(1:3),'per') % 周期性边界条件
    lan=hx(1)/(hx(1)+hx(end)); % 计算λ_n的值
    mun=1-lan; % 计算μ_n的值
    A=2*eye(n-1)+diag(la,1)+diag([mu(2:n-2);mun],-1); % 系数矩阵
    A(1,n-1)=mu(1); % 对系数矩阵指定位置赋值
    A(n-1,1)=lan; % 对系数矩阵指定位置赋值
    b=[d;6*(hy(1)/hx(1)-hy(end)/hx(end))/(hx(1)+hx(end))]; % 右端向量
    M1=A\b; % 利用左除法求Mi (i=2:n)
    M=[M1(n-1);M1]; % 所有节点处的二阶导数值
else
    error('无效的边界条件.') % 无效的边界条件，返回错误信息
end
for k=1:n-1
    c=hy(k)/hx(k)-(1/3*M(k)+1/6*M(k+1))*hx(k); % 一次式的系数
    pp(k,1:4)=[0,0,0,ydata(k)]+... % 构造每个区间的三次多项式，此为零次式
        [0,0,c*poly(xdata(k))]+... % 此为一次式
        [0,1/2*M(k)*poly(xdata([k,k]))]+... % 此为二次式
        1/6/hx(k)*(M(k+1)-M(k))*poly(xdata([k,k,k])); % 此为三次式
    pp(k,5:6)=xdata(k:k+1); % 分段区间
    y=@(x)y(x)+polyval(pp(k,1:4),x).*(x>=xdata(k) & x<xdata(k+1)); % 将每段的插值函数累加
end
if exist('xi','var') % 判断插值点是否存在
    y=feval(y,xi); % 根据求得的插值表达式求指定点的值
    y(xi==xdata(n))=ydata(n); % 若所求点包含xdata的最后一个元素，则将ydata的最后一个元素赋予它
end
pp=array2table(pp,'variablenames',{'Ai','Bi','Ci','Di','xi','xi_1'});
[varargout{1:2}]=deal(y,... % 第1个输出参数为插值多项式或其在某点处的值
    pp); % 第2个输出参数为插值基函数
```

#### （五）示例（例6 - 8）
利用三次样条插值法对函数$f(x)=\frac{1}{1 + 25x^2}$，$x = -1:0.2:1$进行插值。在MATLAB命令窗口中，通过定义函数、节点和插值点，分别使用不同边界条件调用`pcsinterp`函数进行插值，并绘制插值结果和误差曲线，直观展示不同边界条件下三次样条插值的效果。 

## 2.6 MATLAB自带函数应用
#### 2.6.1 polyfit函数
- **功能与调用格式**：MATLAB的`polyfit`函数用于求解多项式插值和拟合，调用格式为`p = polyfit(x,y,n)` ，其中`x`和`y`是节点数据向量，`n`是多项式次数。当`n<length(x)-1`时执行多项式拟合；当`n=length(x)-1`时进行多项式插值；当`n≥length(x)`时会给出警告信息，`p`为返回的多项式系数向量。
- **示例（例6 - 10）**：针对某年全国1 - 8月的股票交易成交金额统计数据，共8个数据点，需用7次多项式插值。在MATLAB命令窗口输入相关语句，先定义月份向量`month`，沪深两市交易额向量`SH`、`SZ`，绘制数据点后，通过`polyfit`函数进行7次多项式拟合，再用`fplot`绘制拟合曲线并添加标签，可直观展示数据的插值逼近情况。

#### 2.6.2 interp1函数
- **功能与调用格式**：该函数用于实现分段低次插值，常见调用格式及说明如下：
    - `vq = interp1(x,v,xq)`：利用分段线性插值对数据点`(x,v)`进行插值，返回插值函数在指定点`xq`处的值。
    - `vq = interp1(x,v,xq,method)`：可利用指定插值方法（如`'linear'`分段线性插值、`'nearest'`最邻近插值等）对数据点`(x,v)`插值，并返回指定点`xq`处的值。
    - `vq = interp1(x,v,xq,method,extrapolation)`：针对超出`x`范围的`xq`分量，可按指定方式（如`'extrap'`用相同插值法外插、或指定常数）执行特殊外插。
 - **示例**：
    - **例6 - 11**：对于某地某天中午12:00 - 13:00的温度数据，这是插值逆问题（由函数值反求插值点）。因线性插值在两插值点间单调，可将温度与时间对调，通过`interp1`对相邻监测点线性插值，找出温度为30℃的时间点，并绘图展示。也可通过构造函数并利用`fzero`函数解方程求解此类逆问题。
    - **例6 - 12**：利用样条插值绘制手掌轮廓。先通过`figure`和`axes`创建并最大化图形窗口、创建坐标系，用`ginput`沿手掌轮廓取点，再引入参数`t`，分别对`t - x`和`t - y`用`interp1`进行样条插值，最后绘制插值曲线并隐藏坐标系。

#### 2.6.3 griddata函数
 - **功能与调用格式**：`griddata`函数用于求解散乱节点插值问题，调用格式为`vq = griddata(x,y,v,xq,yq,method)` ，其中`x,y`和`v`是给定节点数据，`xq`和`yq`通常是规则网格点坐标，`method`为可选插值法，包括`'linear'`（基于三角网格划分的线性插值，默认值）、`'cubic'`（三次插值）、`'natural'`（自然邻近插值）、`'nearest'`（最邻近插值）、`'v4'`（MATLAB4中的算法） 。
 - **示例（例6 - 15）**：利用函数`z = xe^{-x^{2}-y^{2}}`生成一组随机点`(x_k,y_k,z_k)`，调用`griddata`函数对这些点进行散乱节点插值。先设置随机数种子，定义函数，生成随机点坐标和网格数据，然后进行插值，分别绘制插值曲面和误差曲面，对比插值结果与原函数值。 

 ### 2.6.1 polyfit函数示例代码（例6 - 10）
```matlab
% 定义月份向量，1 - 8月
month = 1:8; 
% 假设沪深两市交易额向量（这里是示例数据，需根据实际数据替换）
SH = [100 120 110 130 140 135 150 160]; 
SZ = [80 90 85 95 105 100 110 120]; 

% 绘制沪深两市交易额数据点
figure;
scatter(month, SH, 'ro', 'DisplayName', '沪市交易额');
hold on;
scatter(month, SZ, 'go', 'DisplayName', '深市交易额');

% 7次多项式拟合
p_SH = polyfit(month, SH, 7); 
p_SZ = polyfit(month, SZ, 7); 

% 绘制拟合曲线
x_fit = 1:0.1:8;
y_fit_SH = polyval(p_SH, x_fit);
y_fit_SZ = polyval(p_SZ, x_fit);
fplot(@(x)polyval(p_SH, x), [1 8], 'r--', 'DisplayName', '沪市拟合曲线');
fplot(@(x)polyval(p_SZ, x), [1 8], 'g--', 'DisplayName', '深市拟合曲线');

xlabel('月份');
ylabel('交易额');
title('1 - 8月沪深两市股票交易成交金额插值拟合');
legend;
hold off;
```

### 2.6.2 interp1函数示例代码
#### 例6 - 11代码
```matlab
% 假设中午12:00 - 13:00温度数据（时间为分钟数，从0开始，这里是示例数据，需根据实际数据替换）
time = 0:10:60; 
temperature = [28 29 30.5 31 30 29.5 29]; 

% 将温度与时间对调
t_temperature = temperature;
t_time = time;

% 用interp1进行线性插值找温度为30℃的时间点
t_30 = interp1(t_temperature, t_time, 30); 

% 绘图
figure;
scatter(time, temperature, 'bo', 'DisplayName', '实际温度数据');
hold on;
plot(t_30, 30, 'ro', 'MarkerSize', 10, 'DisplayName', '温度为30℃的时间点');
xlabel('时间（分钟）');
ylabel('温度（℃）');
title('12:00 - 13:00温度数据插值');
legend;
hold off;
```

#### 例6 - 12代码
```matlab
% 创建并最大化图形窗口、创建坐标系
figure('Position', get(0, 'Screensize')); 
ax = axes('Position', [0 0 1 1]);

% 沿手掌轮廓取点
[x, y] = ginput; 

% 引入参数t
t = 1:length(x); 

% 对t - x和t - y用interp1进行样条插值
t_fine = 1:0.1:length(x);
x_interp = interp1(t, x, t_fine,'spline');
y_interp = interp1(t, y, t_fine,'spline');

% 绘制插值曲线
plot(x_interp, y_interp, 'b - ');

% 隐藏坐标系
axis off;
```

### 2.6.3 griddata函数示例代码（例6 - 15）
```matlab
% 设置随机数种子
rng(0); 

% 定义函数
fun = @(x,y) x.*exp(-x.^2 - y.^2); 

% 生成随机点坐标
x = rand(100, 1)*2 - 1; 
y = rand(100, 1)*2 - 1; 
% 计算随机点的函数值
z = fun(x, y); 

% 生成规则网格数据
[xq, yq] = meshgrid(-1:0.1:1, -1:0.1:1); 
% 用griddata进行线性插值
zq_linear = griddata(x, y, z, xq, yq, 'linear'); 
% 用griddata进行三次插值
zq_cubic = griddata(x, y, z, xq, yq, 'cubic'); 

% 绘制插值曲面
figure;
subplot(2, 2, 1);
surf(xq, yq, zq_linear);
title('线性插值曲面');
xlabel('x');
ylabel('y');
zlabel('z');

subplot(2, 2, 2);
surf(xq, yq, zq_cubic);
title('三次插值曲面');
xlabel('x');
ylabel('y');
zlabel('z');

% 计算误差
z_true = fun(xq, yq);
error_linear = abs(z_true - zq_linear);
error_cubic = abs(z_true - zq_cubic);

% 绘制误差曲面
subplot(2, 2, 3);
surf(xq, yq, error_linear);
title('线性插值误差曲面');
xlabel('x');
ylabel('y');
zlabel('误差');

subplot(2, 2, 4);
surf(xq, yq, error_cubic);
title('三次插值误差曲面');
xlabel('x');
ylabel('y');
zlabel('误差');
``` 