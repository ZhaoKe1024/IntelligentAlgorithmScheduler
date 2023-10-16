# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2021-04-22 11:22
import numpy as np
from matplotlib import pyplot as plt
from DPSOTaskScheduling import DPSO
from ChaosDPSOScheduler import ChaosDPSO
from ChaosReproducePSO import ChaosReproductionPSO
from ACScheduler import ACScheduler
from intelligentAlgorithm.IntelligentAlgorithmScheduler.schedulers.GAScheduler import GAScheduler
from utils.dataExamples import get_data_r2n3c12, get_data_r3n4c18
from utils.dataExamples import get_data_r2n5c20, get_data_r2n6c24, get_data_r2n7c28


def select_data(ind: int):
    data = None
    if ind == 0:
        data = get_data_r2n3c12(1)
    elif ind == 1:
        data = get_data_r3n4c18(1)
    elif ind == 2:
        data = get_data_r2n5c20(1)
    elif ind == 3:
        data = get_data_r2n6c24()
    elif ind == 4:
        data = get_data_r2n7c28()
    return data


def iterate(nodes, lets, saveind):
    population = 300
    MAX_GEN = 300

    crpso = ChaosReproductionPSO(lets, nodes, population_number=population, times=MAX_GEN)
    cpso = ChaosDPSO(lets, nodes, population_number=population, times=MAX_GEN)
    dpso = DPSO(cloudlets=lets, vms=nodes, population_number=population, times=MAX_GEN)
    ga = GAScheduler(lets, nodes, population_number=population, times=MAX_GEN)
    aco = ACScheduler(lets, nodes, population_number=population, times=MAX_GEN)

    crpso_generation = crpso.exec()
    cpso_generation = cpso.exec()
    dpso_generation = dpso.exec()
    ga_generation = ga.execute()
    aco_generation = aco.scheduler_main()
    res = [crpso_generation, cpso_generation, dpso_generation, ga_generation, aco_generation]
    np.savetxt("chaos-res-matrix/res-"+str(saveind)+".txt", res, delimiter=',', encoding="GB2312")
    # return crpso_generation[-1], cpso_generation[-1], dpso_generation[-1], ga_generation[-1], aco_generation[-1]


def test(ind: int):
    X = np.arange(5)
    y = np.zeros((5, 5))
    for i in range(5):
        data = select_data(i)
        res = iterate(data["nodes"], data["cloudlets"])
        y[:, i] = res
    np.savetxt("data.txt", y, delimiter=',', encoding="GB2312")
    # # plt.style.use("fivethirtyeight")
    # ax = plt.figure(ind)
    # p1, = plt.plot(range(5), y[0, :])
    # p2, = plt.plot(range(5), y[1, :])
    # p3, = plt.plot(range(5), y[2, :])
    # p4, = plt.plot(range(5), y[3, :])
    # p5, = plt.plot(range(5), y[4, :])
    #
    # ax.legend([p1, p2, p3, p4, p5], ["CRPSO", "CPSO", "DPSO", "GA", "ACO"], bbox_to_anchor=(1.01, 0.88), loc=2,
    #           borderaxespad=0)
    # # plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    # # plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    # # plt.title("云计算任务调度-负载均衡得分图")
    # plt.xlabel("task scale")
    # plt.ylabel("fitness")
    # plt.subplots_adjust(right=1)
    # # plt.savefig('result0pic.png', dpi=300, format='png', bbox_inches='tight')  # bbox_inches="tight"解决X轴时间两个字不被保存的问题
    # # plt.savefig("linePlot.png", dpi=300)
    # plt.show()

#
# def stackPlotTest():
#     y = np.loadtxt("data.txt", delimiter=',')
#     labels = ["CRPSO", "CPSO", "DPSO", "GA", "ACO"]
#     colors = ["red", "green", "orange", "purple", "blue"]
#     order = [2, 4, 3, 1, 0]
#     new_data = []
#     new_data.append(y[order[0]])
#     for i in range(1, 5):
#         new_data.append(y[order[i]] - y[order[i - 1]])
#     new_color = [colors[order[0]], colors[order[1]], colors[order[2]], colors[order[3]], colors[order[4]]]
#     new_labels = [labels[order[0]], labels[order[1]], labels[order[2]], labels[order[3]], labels[order[4]]]
#     plt.stackplot(range(5), new_data[0], new_data[1], new_data[2], new_data[3], new_data[4], colors=new_color,
#                   labels=new_labels)
#     #
#     # plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
#     # plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
#     # plt.xlabel("teak scale")
#     # plt.ylabel("fitness")
#     # plt.savefig('fiveAlgorithm-cnP3T3-'+str(ind)+'.png', dpi=300,
#     #             format='png', bbox_inches='tight')  # bbox_inches="tight"解决X轴时间两个字不被保存的问题
#     plt.show()


# def test():
#     NUM = 10
#     #分别使用五种方法进行运算，将结果存放在data0.txt中
#     for i in range(5):  # 5组数据
#         data = select_data(i)
#         y = np.zeros((NUM, 5))
#         for j in range(NUM):
#             res = iterate(data["nodes"], data["cloudlets"])
#             y[j, :] = res
#         np.savetxt("data"+str(5+i)+".txt", y, delimiter=',', encoding="GB2312")


def boxPlotTest(i: int):
    # 绘制箱型图
    y0 = np.loadtxt("datas/boxplotdata/data" + str(i) + ".txt", delimiter=',')
    labels = ["", "CRPSO", "CPSO", "DPSO", "GA", "ACO"]
    y0 = 1/y0
    # 共5组数据，分别绘制箱型图
    ax = plt.figure(i)
    plt.boxplot((y0[:, 0], y0[:, 1], y0[:, 2], y0[:, 3], y0[:, 4]))
    plt.ylabel("load balance", fontdict={'size': 18})
    plt.xticks(range(6), labels, size=16)
    plt.yticks(size=18)
    plt.show()
    # plt.savefig("boxplot" + str(i) + ".png")
    # plt.close(i)


def LinePlot(ind: int):  # 绘制折线图
    data = np.loadtxt("datas/lineplotdata/data" + str(ind) + ".txt", delimiter=",")
    # plt.style.use("fivethirtyeight")
    ax = plt.figure(ind)

    p0, = plt.plot(range(5), data[:, 0], "blue")
    p1, = plt.plot(range(5), data[:, 1], "orange")
    p2, = plt.plot(range(5), data[:, 2], "green")
    p3, = plt.plot(range(5), data[:, 3], "red")
    p4, = plt.plot(range(5), data[:, 4], "purple")

    # ax.legend([p0, p1, p2, p3, p4], ["CRPSO", "CPSO", "DPSO", "GA", "ACO"], bbox_to_anchor=(1.01, 0.88), loc=2,
    #           borderaxespad=0)
    ax.legend([p0, p1, p2, p3, p4], ["CRPSO", "CPSO", "DPSO", "GA", "ACO"])
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    plt.title("Cloudlets Scheduling Algorithm-Dataset")
    plt.xlabel("task scale")
    plt.ylabel("fitness")
    # plt.subplots_adjust(right=1)
    # plt.savefig('result0pic.png', dpi=300, format='png', bbox_inches='tight')  # bbox_inches="tight"解决X轴时间两个字不被保存的问题
    # plt.savefig("linePlot.png", dpi=300)
    plt.show()


def getOrder(ind: int):
    if ind == 1:
        return [4, 3, 2, 1, 0]
    elif ind == 2:
        return [3, 4, 2, 1, 0]
    elif ind == 3:
        return [3, 4, 2, 1, 0]
    elif ind == 4:
        return [3, 4, 1, 2, 0]
    elif ind == 5:
        return [3, 4, 2, 1, 0]
    elif ind == 6:
        return [3, 4, 1, 2, 0]
    elif ind == 7:
        return [3, 4, 1, 2, 0]


# 1 [4, 3, 1, 2, 0]
# 2 order = [3, 4, 2, 1, 0]
# 4 []
def stackPlotTest(ind: int):
    # 绘制堆叠图
    y = np.loadtxt("datas/lineplotdata/data" + str(ind) + ".txt", delimiter=',')
    labels = ["CRPSO", "CPSO", "DPSO", "GA", "ACO"]
    colors = ["#87CEFA", "#FFD700", "#98FB98", "#FF4500", "#9370DB"]
    order = getOrder(ind)
    new_data = []
    new_data.append(y[:, order[0]])
    for i in range(1, 5):
        new_data.append(y[:, order[i]] - y[:, order[i - 1]])
    new_color = [colors[order[0]], colors[order[1]], colors[order[2]], colors[order[3]], colors[order[4]]]
    new_labels = [labels[order[0]], labels[order[1]], labels[order[2]], labels[order[3]], labels[order[4]]]
    ax = plt.figure(ind)
    plt.stackplot(range(1, 6), new_data[0], new_data[1], new_data[2], new_data[3], new_data[4], colors=new_color,
                  labels=new_labels)
    ax.legend(new_labels)
    # plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    # plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    # plt.title("Cloudlets Scheduling Algorithm-Dataset")
    plt.xlabel("teak scale")
    plt.ylabel("fitness")
    plt.xticks(size=12)
    plt.yticks(size=12)
    # ax.set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])
    # plt.savefig('datas/lineplotdata/stack'+str(ind)+'.png', dpi=300,
    #             format='png', bbox_inches='tight')  # bbox_inches="tight"解决X轴时间两个字不被保存的问题
    plt.show()


if __name__ == '__main__':
    data = select_data(1)
    for i in range(20, 25):
        iterate(data["nodes"], data["cloudlets"], i)
