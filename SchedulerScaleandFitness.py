# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2021-04-27 10:15
import numpy as np
from matplotlib import pyplot as plt
from schedulers.DPSOTaskScheduling import DPSO
from schedulers.HPRPSOScheduler import HPRPSO
from schedulers.SAScheduler import SAScheduler
from intelligentAlgorithm.IntelligentAlgorithmScheduler.schedulers.GAScheduler import GAScheduler
from schedulers.ACScheduler import ACScheduler
from utils.dataExamples import get_data_r2n3c12, get_data_r2n4c18
from utils.dataExamples import get_data_r2n5c20, get_data_r2n6c24, get_data_r2n7c28


def scheduler_test(data):

    print("节点数量：", len(data["nodes"]), ", 任务数量：", len(data["cloudlets"]))
    population = 50
    MAX_GEN = 200

    hprpso = HPRPSO(cloudlets=data["cloudlets"], vms=data["nodes"], population_number=population, times=MAX_GEN)
    dpso = DPSO(cloudlets=data["cloudlets"], vms=data["nodes"], population_number=population, times=MAX_GEN)
    sa = SAScheduler(cloudlets=data["cloudlets"], vms=data["nodes"], population_number=population, times=MAX_GEN)
    ac = ACScheduler(cloudlets=data["cloudlets"], vms=data["nodes"], population_number=population, times=MAX_GEN)
    ga = GAScheduler(cloudlets=data["cloudlets"], vms=data["nodes"], population_number=population, times=MAX_GEN)

    hprpso_generation = hprpso.exec()
    dpso_generation = dpso.execute()
    sa_generation = sa.execute()
    ac_generation = ac.scheduler_main()
    ga_generation = ga.execute()

    return [hprpso_generation[-1], dpso_generation[-1], sa_generation[-1], ac_generation[-1], ga_generation[-1]]


def select_data(ind: int):
    data = None
    if ind == 0:
        data = get_data_r2n3c12(1)
    elif ind == 1:
        data = get_data_r2n4c18(1)
    elif ind == 2:
        data = get_data_r2n5c20(1)
    elif ind == 3:
        data = get_data_r2n6c24()
    elif ind == 4:
        data = get_data_r2n7c28()
    return data


def different_data_scheduler():
    y = np.zeros((5, 5))

    for i in range(5):
        data = select_data(i)
        res = scheduler_test(data)
        y[:, i] = res

    np.savetxt("result0dat.txt", y, delimiter=",", fmt="%1.4e", newline=",\n")

    ax = plt.figure(0)
    p1, = plt.plot(range(5), y[0, :])
    p2, = plt.plot(range(5), y[1, :])
    p3, = plt.plot(range(5), y[2, :])
    p4, = plt.plot(range(5), y[3, :])
    p5, = plt.plot(range(5), y[4, :])

    ax.legend([p1, p2, p3, p4, p5], ["HPRPSO", "DPSO", "SA", "ACO", "GA"], bbox_to_anchor=(1.01, 0.88), loc=2, borderaxespad=0)
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    # plt.title("云计算任务调度-负载均衡得分图")
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.subplots_adjust(right=1)
    plt.savefig('imgs/stackPlot/result0pic.png', dpi=300, format='png', bbox_inches='tight')  # bbox_inches="tight"解决X轴时间两个字不被保存的问题
    # plt.show()


if __name__ == '__main__':
    different_data_scheduler()

