# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2021-03-20 17:50
import numpy as np
from utils.Entities import Cloudlet, VM, calculate_fitness
import matplotlib.pyplot as plt


class SAScheduler:
    def __init__(self, cloudlets, vms, population_number=100, times=500):
        self.cloudlets = cloudlets
        self.vms = vms
        self.cloudlet_num = len(cloudlets)  # 任务数量也就是粒子长度
        self.machine_number = len(vms)  # 机器数量

        self.population_number = population_number  # 种群数量
        self.times = times  # 迭代代数

        # 初始温度
        self.T0 = 1000
        # 迭代温度
        self.T = self.T0
        # 每个温度下的迭代次数
        self.Lk = 100
        # 温度衰减系数
        self.alpha = 0.95

        self.best_bl = None
        self.best_way = None

    # 生成新解, 倒置部分解向量
    def gen_new_way(self, way0):
        index1 = np.random.randint(1, self.cloudlet_num)
        index2 = np.random.randint(1, self.cloudlet_num)
        while np.abs(index2 - index1) < 3:
            index2 = np.random.randint(1, self.cloudlet_num)
        way1 = way0.copy()
        way1[index1: index2: 1] = way1[(index2 - 1): (index1 - 1): -1]
        return way1    # 生成新解, 倒置部分解向量

    # 交换两点
    def gen_new_way_change(self, way0):
        index1 = np.random.randint(1, self.cloudlet_num)
        index2 = np.random.randint(1, self.cloudlet_num)
        way1 = way0.copy()
        way1[index1], way1[index2] = way1[index2], way1[index1]
        return way1

    def sa_main(self):
        self.best_way = np.random.randint(0, self.machine_number, self.cloudlet_num)
        # temp_solution = [0] * self.times
        # temp_value = [0] * self.times

        results = list()
        for t in range(self.times):
            if t % 20 == 0:
                print("SA Iter, ", t, "/", self.times, ", 适应度：", self.best_bl)

            # for j in range(self.Lk):
            for j in range(self.population_number):
                self.best_bl = calculate_fitness(self.best_way, self.cloudlets, self.vms)
                way1 = self.gen_new_way(self.best_way)
                way1 = self.gen_new_way_change(way1)
                # print("新解的适应度:", bl1)
                bl1 = calculate_fitness(way1, self.cloudlets, self.vms)
                if bl1 > self.best_bl:
                    self.best_way = way1
                    # if bl1 > temp_value[j]:
                    #     temp_solution[j] = way1
                    #     temp_value[j] = bl1
                else:
                    p = np.exp((bl1 - self.best_bl) / self.T)
                    if np.random.rand() < p:
                        self.best_way = way1
            results.append(self.best_bl)
            # results.append(temp_value[t])
            self.T = self.alpha * self.T
        #     print("最佳适应度", self.best_bl)
        #     print("最佳路径", self.best_way)
        # print("最佳适应度：", self.best_bl)
        # print("策略：", self.best_way)

        return results


if __name__ == '__main__':
    nodes = [
        VM(0, 0.762, 2, 920, 2223, 400, 2000, 5.9, 30),
        VM(1, 0.762, 2, 1200, 2223, 1000, 2000, 6, 30),
        VM(2, 0.762, 2, 850, 2223, 800, 2000, 5.8, 30),
        VM(3, 0.762, 2, 1200, 2223, 900, 2000, 5.9, 30),  # 4
    ]
    lets = [
        Cloudlet(0.078400, 60.689797, 228.9767525518272, 2.712677828249846),
        Cloudlet(0.065683, 185.848012, 187.97925460500625, 5.1178778788024),
        Cloudlet(0.050440, 96.030497, 206.77315938787453, 4.264445831060432),
        Cloudlet(0.104019, 131.428883, 218.78608382384854, 2.209277743955084),  # 4
        Cloudlet(0.022355, 192.582491, 231.9710696727387, 3.26584657336946),
        Cloudlet(0.232862, 226.085299, 233.03395445541793, 4.289629843497603),
        Cloudlet(0.194654, 77.503350, 190.41556439297744, 4.626189837323374),
        Cloudlet(0.148194, 241.349622, 264.54311244786555, 4.095493414214854),  # 8
        Cloudlet(0.146926, 199.978750, 248.2824412513349, 3.6236622746002953),
        Cloudlet(0.081256, 149.824589, 243.16971522421468, 4.009965930243791),
        Cloudlet(0.237547, 141.050771, 277.01199985466394, 4.671274901135505),
        Cloudlet(0.138457, 139.508608, 271.25359518569496, 3.9828754698861477),
        Cloudlet(0.088451, 133.618232, 245.98393640211285, 3.81448563152322),
        Cloudlet(0.266167, 156.087665, 214.0395006818089, 5.657246768827748),
        Cloudlet(0.130581, 158.033508, 251.24327206708733, 5.252957834065088),
        Cloudlet(0.099247, 211.409329, 197.81288865451026, 4.240369159034978),  # 16
        Cloudlet(0.124647, 259.696868, 245.59672377663492, 7.850605743087694),
        Cloudlet(0.076976, 186.666789, 277.3108057619953, 2.440325446644967),  # 18
    ]
    sa = SAScheduler(lets, nodes, population_number=500,  times=500)
    data = sa.sa_main()
    i = 0
    for _ in sa.best_way:
        print("任务:", i, " 放置到机器", sa.vms[sa.best_way[i]].id, "上执行")
        i += 1
    plt.plot(range(sa.times), data)  # 正常应该是2.7左右
    # plt.savefig('imgr2/BPOScheduler-0.95_2_2--vmax5-popu100-iter200-w095-cg2-cl2.png', dpi=300,
    #             format='png')  # bbox_inches="tight"解决X轴时间两个字不被保存的问题
    plt.show()
