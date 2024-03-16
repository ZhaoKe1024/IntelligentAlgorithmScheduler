# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2021-03-19 15:22
import numpy as np
from eautils.Entities import Cloudlet, VM, calculate_fitness
from matplotlib import pyplot as plt


class ACScheduler:
    def __init__(self, cloudlets, vms, population_number=100, times=500):
        self.cloudlets = cloudlets
        self.vms = vms
        self.cloudlet_num = len(cloudlets)  # 任务数量也就是粒子长度
        self.machine_number = len(vms)  # 机器数量

        self.population_number = population_number  # 种群数量
        self.times = times  # 迭代代数

        # 表示任务选择的机器的信息素
        self.topo_phs = [[100 for _ in range(self.machine_number)] for _ in range(self.cloudlet_num)]
        # 最优策略
        self.best_topo = None

    # 生成新的解向量--根据信息素浓度生成
    def gen_topo_jobs(self):
        ans = [-1 for _ in range(self.cloudlet_num)]
        node_free = [node_id for node_id in range(self.machine_number)]
        for let in range(self.cloudlet_num):
            ph_sum = np.sum(list(map(lambda j: self.topo_phs[let][j], node_free)))
            test_val = 0
            rand_ph = np.random.uniform(0, ph_sum)
            for node_id in node_free:
                test_val += self.topo_phs[let][node_id]
                if rand_ph <= test_val:
                    ans[let] = node_id
                    break
        return ans

    # 更新信息素
    def update_topo(self):
        for i in range(self.cloudlet_num):
            for j in range(self.machine_number):
                if j == self.best_topo[i]:
                    self.topo_phs[i][j] *= 2
                else:
                    self.topo_phs[i][j] *= 0.5


    def scheduler_main(self):
        results = [0 for _ in range(self.times)]
        fitness = 0

        for it in range(self.times):
            best_time = 0

            for ant_id in range(self.population_number):
                topo_jobs = self.gen_topo_jobs()
                fitness = calculate_fitness(topo_jobs, self.cloudlets, self.vms)
                if fitness > best_time:
                    self.best_topo = topo_jobs
                    best_time = fitness
            assert self.best_topo is not None
            self.update_topo()
            results[it] = best_time
            if it % 20 == 0:
                print("ACO iter: ", it, " / ", self.times, ", 适应度: ", fitness)
        plt.plot(range(self.times), results)
        plt.xlabel("迭代次数")
        plt.ylabel("适应度")
        plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        plt.title("蚁群算法求解云任务调度负载均衡问题")
        # plt.savefig('img/ACScheduler-1.1_0.9-popu100-iter200.png', dpi=300,
        #             format='png')  # bbox_inches="tight"解决X轴时间两个字不被保存的问题
        plt.show()
        return results


if __name__ == '__main__':
    # 第四组数据data9

    nodes = [
        VM(0, 0.662, 2, 620, 2223),
        VM(1, 0.662, 2, 1100, 2223),
        VM(2, 1.662, 2, 720, 2223),
        VM(3, 0.662, 2, 1100, 2223),
        VM(4, 0.662, 2, 620, 2223),
        VM(5, 0.562, 2, 650, 2223),
        VM(6, 0.562, 2, 620, 2223),
        VM(7, 0.462, 2, 440, 2223),  # 8
    ]
    lets = [
        Cloudlet(0.133364, 272.435810),
        Cloudlet(0.226357, 141.126392),
        Cloudlet(0.084122, 7.183883),
        Cloudlet(0.029290, 96.658838),
        Cloudlet(0.027560, 247.821058),
        Cloudlet(0.191912, 80.636804),
        Cloudlet(0.134658, 220.702279),
        Cloudlet(0.133052, 163.046071),
        Cloudlet(0.272010, 253.477271),
        Cloudlet(0.175000, 19.409176),
        Cloudlet(0.166933, 140.880123),
        Cloudlet(0.286495, 71.288800),
        Cloudlet(0.080714, 354.839232),
        Cloudlet(0.209842, 211.351191),
        Cloudlet(0.221753, 249.500490),
        Cloudlet(0.128952, 81.599575),
        Cloudlet(0.168469, 122.216016),
        Cloudlet(0.049628, 135.728968),
        Cloudlet(0.051167, 230.172949),
        Cloudlet(0.158938, 135.356776),
        Cloudlet(0.212047, 202.830773),
        Cloudlet(0.372328, 13.145747),
        Cloudlet(0.092549, 130.122476),
        Cloudlet(0.166031, 97.761267),
        Cloudlet(0.142820, 45.852985),
        Cloudlet(0.016367, 189.495519),
        Cloudlet(0.112156, 173.926518),
        Cloudlet(0.004466, 156.806505),
        Cloudlet(0.222208, 62.619918),
        Cloudlet(0.073526, 232.175486),
        Cloudlet(0.158527, 178.624649),
        Cloudlet(0.103075, 133.896667),
        Cloudlet(0.176026, 156.076929),
        Cloudlet(0.098275, 136.450544),
        Cloudlet(0.192065, 33.923922),
        Cloudlet(0.213519, 134.820860),

    ]

    ac = ACScheduler(lets, nodes, times=150)
    res = ac.scheduler_main()
    i = 0
    for _ in ac.best_topo:
        print("任务:", i, " 放置到机器", ac.best_topo[i], "上执行")
        i += 1

