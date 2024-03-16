#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf-8 -*-
# @Time : 2021-02-14 23:53
# @Author : ZhaoKe
# @File : test_functions.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab


def DeJong_F1(x):
    # 简单的平方和函数
    # 向量x的长度为3
    # 向量x中元素的范围是[-5.12, 5.12]
    return np.sum(np.clip(x, -5.12, 5.12) * np.clip(x, -5.12, 5.12))


# 又叫Rosenbrock函数
def DeJong_F2(x):
    # 二维函数，具有全局最小值点f(1, 1) = 0
    # 向量x的长度为2
    # 向量x中元素的范围是[-2.048, 2.048]
    return 100 * (x[0]*x[0] - x[1])* (x[0]*x[0] - x[1]) + (1 - x[0])*(1 - x[0])


def DeJong_F3(x):
    # 不连续函数，对于x \in [-5.12, -5.0]中每一个点都取最小值-30
    # 向量x的长度为5
    # 向量x中元素的范围是[-5.12, 5.12]
    # print(np.sum(list(map(np.floor, [5.25, 6.25]))))  # 11
    # print(np.sum(list(map(np.floor, [-5.25, -7.24]))))  # -14
    return np.sum(list(map(int, x)))


def DeJong_F4(x):
    # 含高斯噪声的四次函数，不考虑噪声影响具有全局最小值0
    # 向量x的长度为30
    # 向量x中元素的范围是[-1.28， 1.28]
    # 验证函数编写正确：print(np.sum(np.arange(1, 6) * np.power([1, 2, 3, 4, 5], 4)))  # 4425
    # 1+2*16+3*81+4*256+5*625 = 1+32+243+1024+3125=1+275+4149=276+4149=4425
    return np.sum(np.arange(1, 31) * np.power(x, 4)) + np.random.randn()


# 有bug
def DeJong_F5(x):
    # 多峰值函数，有25个局部极小值点
    # 向量x的长度为2
    # 向量x中元素的范围是[-65.536, 65.536]
    # 全局最小值f(-32, -32) = 0.998
    # print(DeJong_F5([-32, -32]))  # 1.2896193156872444 ，不知道问题在哪里
    a = list()
    a.append([-32 + x * 16 for x in range(5)] * 5)
    b = []
    temp = -48
    for i in range(25):
        if i % 5 == 0:
            temp += 16
        b.append(temp)
    a.append(b)
    a = np.array(a)
    res = 0.002
    for j in range(25):
        temp = 0
        for i in range(2):
            print(x[i] - a[i, j])
            temp += np.power(x[i] - a[i, j], 6)
        res += 1 / (j+1 + temp)
    return res


# # J.D.Schaffer函数
# def Schaffer_F6(x):
#     # 向量x的长度为2
#     # 向量x中元素的范围是[-100， 100]
#     # 只有一个全局极小值点f(0, 0) = 0
#     return 0.5 + (np.sin(np.sqrt(x[0]*x[0] + x[1]*x[1]))*np.sin(np.sqrt(x[0]*x[0] + x[1]*x[1]))-0.5)/((1+0.001*(x[0]*x[0] + x[1]*x[1]))*(1+0.001*(x[0]*x[0] + x[1]*x[1])))


# J.D.Schaffer函数
def Schaffer_F6(x, y):
    # 向量x的长度为2
    # 向量x中元素的范围是[-100， 100]
    # 只有一个全局极小值点f(0, 0) = 0
    return 0.5 + (np.sin(np.sqrt(x*x + y*y))*np.sin(np.sqrt(x*x + y*y))-0.5)/((1+0.001*(x*x + y*y))*(1+0.001*(x*x + y*y)))


def Schaffer_F7(x):
    # 向量x的长度为2
    # 向量x中元素的范围是[-100， 100]
    # 只有一个全局极小值点f(0, 0) = 0
    return np.power(x[0]*x[0] + x[1]*x[1], 0.25)*(np.sin(50*np.power((x[0]*x[0] + x[1]*x[1]), 0.1))*np.sin(50*np.power((x[0]*x[0] + x[1]*x[1]), 0.1))+1)


def Goldstein_Price(x):
    # 向量x的长度为2
    # 向量x中元素的范围是[-2，2]
    # 只有一个全局极小值点f(0, -1) = 3
    return (1+(x[0]+x[1]+1)*(x[0]+x[1]+1)*(19-14*x[0]+3*x[0]*x[0]-14*x[1]+6*x[0]*x[1]+3*x[1]*x[1]))*(30+(2*x[0]-3*x[1])*(2*x[0]-3*x[1])*(18-32*x[0]+12*x[0]*x[0]+48*x[1]-36*x[0]*x[1]+27*x[1]*x[1]))


def Shubert(x, y):
    # 向量x的长度为2
    # 向量x中元素的范围是[-10，10]
    # 总共有760个局部极小值点，其中全局极小值点有18个，值为f = 186.731

    # print(Shubert([0, 0]))  # 19.875836249802127
    # print(Shubert([1, -1]))  # -14.453253529290407
    # print((np.cos(1)+2*np.cos(2)+3*np.cos(3)+4*np.cos(4)+5*np.cos(5))*(np.cos(1)+2*np.cos(2)+3*np.cos(3)+4*np.cos(4)+5*np.cos(5)))  # 19.875836249802127
    # print((np.cos(3)+2*np.cos(5)+3*np.cos(7)+4*np.cos(9)+5*np.cos(11))*(np.cos(1)+2*np.cos(1)+3*np.cos(1)+4*np.cos(1)+5*np.cos(1)))  # -14.453253529290407
    # NL = np.arange(1, 6)
    # NL = list(range(1, 6))
    res_1 = 0
    res_2 = 0
    for i in range(1, 6):
        res_1 += i * np.cos((i+1) * x + i)
    for j in range(1, 6):
        res_2 += j * np.cos((j + 1) * y + j)
    return res_1 * res_2


def SixHumpCamelBackFunction(x, y):
    # 向量x的长度为2
    # 元素x[0]的范围是[-3，3]
    # 元素x[1]的范围是[-2, 2]
    # 一共6个局部极小值点。其中两个全局最小值点
    # f(-0.0898, 0.7126)=f(0.0898, -0.7126) = -1.031628
    # print(SixHumpCamelBackFunction([-0.0898, 0.7126]))
    # print(SixHumpCamelBackFunction([0.0898, -0.7126]))
    # return (4-2.1*x[0]*x[0]+x[0]*x[0]*x[0]*x[0]/3)*x[0]*x[0] + x[0]*x[1] + (-4+4*x[1]*x[1])*x[1]*x[1]
    return (4-2.1*x*x+x*x*x*x/3)*x*x + x*y + (-4+4*y*y)*y*y


# n维函数
def Restrigin(x):
    # [-5.12, 5.12]
    # 最优值 零点的零
    x = np.array(x)
    return np.sum(x * x - 10 * np.cos(2*np.pi*x) + 10)


def Griewank(x):
    # [-600, 600]
    # 最优值 零点的零
    # print(Griewank([0.1, 0.1, 0.1]))
    # print(Griewank([0,0,0]))
    x = np.array(x)
    res = 1 / 4000 * np.sum(x * x)
    res_1 = 1
    for i in range(len(x)):
        res_1 *= np.cos(x[i] / (i+1))
    return res - res_1 + 1


def mesh_plot():
    DELTA = 0.01
    x = np.arange(-3.0, 3.0, DELTA)
    y = np.arange(-2.0, 2.0, DELTA)
    X, Y = np.meshgrid(x, y)
    # print('x：')
    # print(X)
    # print('y: ')
    # print(Y)
    Z1 = DeJong_F4(X, Y)
    # fig = plt.figure()
    ax = plt.gca(projection='3d')
    ax.plot_surface(X, Y, Z1, cmap=plt.get_cmap('rainbow'), linewidth=0.2)
    # ax.scatter(X, Y, Z1)
    plt.show()


# 带有复杂约束的函数省略


if __name__ == '__main__':
    mesh_plot()
    # print(DeJong_F5([-32, -32]))  # 1.2896193156872444 ，不知道问题在哪里
