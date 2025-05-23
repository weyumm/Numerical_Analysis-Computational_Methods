﻿# 第一章 误差与有效数字

## 1.1 误差分类
### （一）模型误差
在用计算机解决实际问题时，需将实际问题转化为数学模型。数学模型是对实际问题的抽象、简化，忽略了一些次要因素，是客观现象的近似、粗糙描述。数学模型与实际问题之间出现的误差称为模型误差。例如牛顿第二定律建立的模型，虽为优秀近似，但实际中因风、空气阻力等微小变化因素，无法准确预测结果，计算时需进行处理或简化。

### （二）参数误差（观测误差）
数学模型中物理参数的具体数值一般通过实验测定或观测得到，与真值之间存在的误差称为参数误差或观测误差。产生原因如下：
1. **测量仪器**：测量依赖测量仪器，每种仪器精度有限，且受制造工艺限制存在一定误差，限制了观测值精度。
2. **观测者**：不同观测者感觉器官辨别能力有差异，同一观测者在不同时间、空间的辨别能力也会不同。 
3. **外界条件**：温度、湿度、压强、风力、大气折光、电离层等因素会直接影响观测结果，且随这些因素变化影响不同，导致观测结果产生误差。

### （三）方法误差（截断误差）
在数学模型及其参数值确定后，采用数值方法计算时，由于多数数值方法是近似方法，计算结果与准确值之间存在的误差，称为方法误差或截断误差。

**示例**：对于定积分$I = \int_{0}^{\frac{1}{2}}\frac{1}{1 + x^{3}}dx$ 
1. **解析解求解**：在MATLAB中，使用符号运算工具箱的`int`函数可求其解析解。
2. **近似值计算**：利用5阶泰勒级数近似代替被积函数计算近似值，需使用`taylor`函数进行泰勒展开。

**MATLAB实现代码**：
```matlab
syms x; % 声明符号变量
y = 1/(1 + x^3); % 被积函数
I1 = int(y, 0, 1/2); % 定积分的解析解
yt = taylor(y, x, 'order', 6); % 被积函数5阶泰勒展开
I2 = int(yt, 0, 1/2); % 泰勒展开式的定积分
fplot(y, [0, 1/2], 'k', 'linewidth', 2); % 绘制被积函数
hold on; % 图形保持
fplot(yt, [0, 1/2], 'k--', 'linewidth', 2); % 绘制被积函数的5阶泰勒逼近多项式
legend(gca, {'$y =' + latex(y) + '$', '${y_t}=' + latex(yt) + '$'}, 'interpreter', 'latex', 'fontsize', 12, 'Position', [0.65, 0.7, 0.25, 0.2]); % 添加图例
text(0.05, 0.92, {'$\int_{0}^{\frac{1}{2}} {y}dx=' + latex(I1) + '$'}, 'interpreter', 'latex', 'fontsize', 12); % 添加文本标注
text(0.05, 0.9, {'$\int_{0}^{\frac{1}{2}} {y_t}dx=' + latex(I2) + '$'}, 'interpreter', 'latex', 'fontsize', 12); % 添加文本标注
```

### （四）舍入误差
在计算机数值计算中，因数据位数可能很多甚至无穷，而计算机字长有限，对中间结果数据按“四舍五入”等规则取近似值，导致计算过程产生的误差，称为舍入误差。例如，在MATLAB中，$\left[-\frac{9007199254740993}{9007199254740992}, -\frac{18014398509481983}{18014398509481984}\right)$内的所有实数都表示为 -1，就会产生舍入误差。

### 误差分类总结
 - **固有误差**：建立数学模型时就存在的模型误差和观测误差，是客观存在的。
 - **计算误差**：由计算方法引起的截断误差和舍入误差，在数值计算方法中主要讨论计算误差。 

## 1.2 绝对误差与相对误差
### （一）绝对误差
数值误差源于近似的数值操作，准确值$x$与近似值$x^{*}$的差值$e = x - x^{*}$，称为近似值$x^{*}$的绝对误差。通常难以算出准确值$x$和误差准确值，只能估计误差范围，即$|e| = |x - x^{*}| \leq \varepsilon$，$\varepsilon$为近似值$x^{*}$的绝对误差限，也可表示为$x = x^{*} \pm \varepsilon$。

### （二）相对误差
绝对误差定义未考虑被估计值量级。例如$x_1 = x_1^{*} \pm e_1 = 10 \pm 1$，$x_2 = x_2^{*} \pm e_2 = 1000 \pm 5$，虽$e_2 > e_1$，但不能说明$x_1^{*}$近似程度比$x_2^{*}$好。因此将误差相对真值归一化，$e_r = \frac{x - x^{*}}{x}$为近似值$x^{*}$的相对误差，相对误差限记为$|e_r| = \left|\frac{x - x^{*}}{x}\right| = \frac{|e|}{|x|} \leq \varepsilon_r$ ，实际中常用$\varepsilon_r^{*} = \frac{\varepsilon}{|x^{*}|}$近似相对误差限。

## 1.3 有效数字
### （一）定义
若近似值$x^{*}$的误差限是某一数位的半个单位，且该位到$x^{*}$的第一位非零数字共有$n$位，则称$x^{*}$具有$n$位有效数字。科学计数法表示为$x^{*} = \pm 0.a_1a_2\cdots a_n \times 10^m$（即$x^{*} = \pm (a_1 \times 10^{-1} + a_2 \times 10^{-2} + \cdots + a_n \times 10^{-n}) \times 10^m$ ），其中$m$为整数，$a_1,a_2,\cdots,a_n$是$0 \sim 9$间整数，且$a_1 \neq 0$，误差限$\varepsilon = \frac{1}{2} \times 10^{m - n}$。

### （二）MATLAB 实现
可编写函数`getdigits`求近似值的有效数字位数及误差限，代码如下：
```matlab
function [n,e]=getdigits(xtrue,x)
% GETDIGITS 获取近似值的有效数字位数及误差限
err=xtrue-x; % 求误差
[~,m]=enotation(x); % 用科学计数法表示近似值
[err,q]=enotation(err); % 用科学计数法表示误差
if err<5 % 判断误差的第一位非零数字是否小于5
    n=m-q; 
else
    n=m-q-1; 
end
e=sym(1/2)*10^(m-n); % 返回误差限
% 利用科学计数法表示数字x，将数字x化为a0.a1a2...*10^m次的形式
    function [x,m]=enotation(x)
        x=abs(x); % 对x取绝对值
        p=0; % 计数器
        while x>=10 % 判断数字x的绝对值是否不小于10
            x=x/10; % 逐步除10，最终得到a0
            p=p+1; % 计数器加1
        end
        if x~=0
            while x<1 % 判断x是否真小数
                x=x*10; % 逐步乘10，最终得到a0
                p=p-1; % 计数器减1
            end
        end
        m=p+1; 
    end
end
```

### （三）示例
已知圆周率$\pi$真值为$3.1415926535897931\cdots$ ：
 - 当近似值$\pi^{*} = 3.1415$时，调用`getdigits`函数可得有效数字位数为$4$，误差限为$1/2000$。
 - 当近似值$\pi^{*} = 3.141593$ 时，调用`getdigits`函数可得有效数字位数为$6$，误差限为$1/200000$ 。 

## 1.4 误差的传播与估计
在数值计算中，参与运算的数据多为近似数，本身带有误差，这些误差会在运算中传播与积累，从而影响计算结果的准确性。虽然精确确定计算结果的精度较为困难，但对计算误差进行定量估计是可行的。

1. **原始公式推导**：
   - 对于函数$f(x_1,x_2)$，其泰勒展开式包含高阶项$+\frac{1}{2!}\left[\left(\frac{\partial^{2}f}{\partial x_{1}^{2}}\right)^{*}\cdot(x_1 - x_{1}^{*})^{2}+2\left(\frac{\partial^{2}f}{\partial x_{1}\partial x_{2}}\right)^{*}\cdot(x_1 - x_{1}^{*})(x_2 - x_{2}^{*})+\left(\frac{\partial^{2}f}{\partial x_{2}^{2}}\right)^{*}\cdot(x_2 - x_{2}^{*})^{2}\right]+\cdots$，其中$x_1 - x_{1}^{*}=e(x_1)$和$x_2 - x_{2}^{*}=e(x_2)$一般为小量值。
   - 忽略高阶小量后，$f(x_1,x_2)$可简化为$f(x_1,x_2)=f(x_{1}^{*},x_{2}^{*})+\left[\left(\frac{\partial f}{\partial x_{1}}\right)^{*}\cdot e(x_1)+\left(\frac{\partial f}{\partial x_{2}}\right)^{*}\cdot e(x_2)\right]$。
2. **绝对误差传播公式**：
   - 设$y = f(x_1,x_2)$，$y^{*}=f(x_{1}^{*},x_{2}^{*})$，绝对误差$e(y)=y - y^{*}$。
   - 则$e(y)=f(x_1,x_2)-f(x_{1}^{*},x_{2}^{*})\approx\left(\frac{\partial f}{\partial x_{1}}\right)^{*}\cdot e(x_1)+\left(\frac{\partial f}{\partial x_{2}}\right)^{*}\cdot e(x_2)$ 。
   - 式中$\left(\frac{\partial f}{\partial x_{1}}\right)^{*}$和$\left(\frac{\partial f}{\partial x_{2}}\right)^{*}$分别是$x_{1}^{*}$和$x_{2}^{*}$对$y^{*}$的绝对误差增长因子，用于表示绝对误差$e(x_1)$和$e(x_2)$经过传播后增大或缩小的倍数。 

3. **相对误差传播公式**
相对误差传播公式为 
$$ \varepsilon_{r}^{*}(y)=\frac{\varepsilon(y)}{\left|y^{*}\right|} \approx \sum_{i = 1}^{n}\left[\left|\left(\frac{\partial f}{\partial x_{i}}\right)^{*}\right| \cdot \frac{\varepsilon\left(x_{i}\right)}{\left|y^{*}\right|}\right]=\sum_{i = 1}^{n}\left[\left|\frac{x_{i}^{*}}{y^{*}}\left(\frac{\partial f}{\partial x_{i}}\right)^{*}\right| \cdot \varepsilon_{r}^{*}\left(x_{i}\right)\right]$$
 。当误差增长因子的绝对值很大时，数据误差经运算传播后，可能导致结果出现较大误差。

### 两数和、差、积与商的误差传播公式
- 绝对误差：
    - $e(x_1 \pm x_2) \approx e(x_1) \pm e(x_2)$
    - $e(x_1x_2) \approx x_2^{*}e(x_1)+x_1^{*}e(x_2)$
    - $e\left(\frac{x_1}{x_2}\right) \approx \frac{1}{x_2^{*}}e(x_1)-\frac{x_1^{*}}{(x_2^{*})^2}e(x_2) \ (x_2^{*} \neq 0)$ 
- 相对误差：
    - $e_{r}^{*}(x_1 \pm x_2) \approx \frac{x_1^{*}}{x_1^{*} \pm x_2^{*}}e_{r}^{*}(x_1) \pm \frac{x_2^{*}}{x_1^{*} \pm x_2^{*}}e_{r}^{*}(x_2)$
    - $e_{r}^{*}(x_1x_2) \approx e_{r}^{*}(x_1)+e_{r}^{*}(x_2)$
    - $e_{r}^{*}\left(\frac{x_1}{x_2}\right) \approx e_{r}^{*}(x_1)-e_{r}^{*}(x_2) \ (x_2 \neq 0)$ 

### 示例
已知$a = \sqrt{2018}$，$b = \sqrt{2017}$ ，要估计$a - b$的有效数字位数。
- **分析思路**：
    - 利用`vpa`函数结合`char`函数和`str2double`函数获取$a$和$b$具有8位有效数字的近似值。
    - 调用`getdigits`函数求得$a$和$b$近似值的误差限。
    - 根据误差传播公式（两数差的绝对误差公式$e(x_1 - x_2) \approx e(x_1)+e(x_2)$ ）得到$a - b$的误差限。
    - 依据误差限反推出$a - b$的有效数字位数。
- **完整MATLAB代码**：
```matlab
% 设置数字显示方式
format longG; 
% 计算a和b的精确值
a = sqrt(2018); 
b = sqrt(2017); 
% 获取a的具有8位有效数字的近似值
a1 = str2double(char(vpa(a, 8))); 
% 获取a1的有效数字位数和误差限
[n1,e1]=getdigits(a,a1); 
% 获取b的具有8位有效数字的近似值
b1 = str2double(char(vpa(b, 8))); 
% 获取b1的有效数字位数和误差限
[n2,e2]=getdigits(b,b1); 
% 计算近似值a1 - b1的最大误差限
e = e1 + e2; 
% 计算科学表示式 x = ±0.a1a2…an×10^m 中m的值
m = ceil(log10(abs(a1 - b1))); 
% 计算近似值有效数字的最小位数，由不等式1/2×10^(m - n)≤e反推得到
n = fix(log10(10^m/(2*e))); 
```
- **结果说明**：
通过上述代码，可得到$a$和$b$近似值的相关信息以及$a - b$的有效数字位数。需要注意，由误差估计式得出绝对误差限和相对误差限时，因取绝对值并用三角不等式放大，是按最坏情形给出，结果较为保守 。 