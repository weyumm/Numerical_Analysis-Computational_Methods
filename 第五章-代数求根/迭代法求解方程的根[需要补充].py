#迭代法

import numpy as np
import math
N0=int(input("请输入最大迭代次数"))
e=float(input("请输入精度"))
x1=float(input("请输入x1："))

def g(x):  #定义转化后的函数
    return math.sin(x)+0.5
k=1
while k<=N0:
    x=g(x1)
    if abs(x-x1)<e:
        print("方程的解为：%s，迭代次数为：%s"%(x,k))
        break
    else:
        x1=x
    k+=1
if k==N0+1:
    print("在%s次迭代之后无法达到所要求的精度"%N0)
