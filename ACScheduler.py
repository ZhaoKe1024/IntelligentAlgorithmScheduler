# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2021-03-19 15:22
import numpy as np
from utils.Entities import Cloudlet, VM, calculate_fitness
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
        return results


# if __name__ == '__main__':
#     # 第四组数据data9
#     nodes = [
#         VM(0, 0.762, 2, 920, 2223, 400, 2000, 5.9, 30),
#         VM(1, 0.762, 2, 1200, 2223, 1000, 2000, 6, 30),
#         VM(2, 0.762, 2, 850, 2223, 800, 2000, 5.8, 30),
#         VM(3, 0.762, 2, 1200, 2223, 900, 2000, 5.9, 30),  # 4
#     ]
#     lets = [
#         Cloudlet(0.078400, 60.689797, 228.9767525518272, 2.712677828249846),
#         Cloudlet(0.065683, 185.848012, 187.97925460500625, 5.1178778788024),
#         Cloudlet(0.050440, 96.030497, 206.77315938787453, 4.264445831060432),
#         Cloudlet(0.104019, 131.428883, 218.78608382384854, 2.209277743955084),  # 4
#         Cloudlet(0.022355, 192.582491, 231.9710696727387, 3.26584657336946),
#         Cloudlet(0.232862, 226.085299, 233.03395445541793, 4.289629843497603),
#         Cloudlet(0.194654, 77.503350, 190.41556439297744, 4.626189837323374),
#         Cloudlet(0.148194, 241.349622, 264.54311244786555, 4.095493414214854),  # 8
#         Cloudlet(0.146926, 199.978750, 248.2824412513349, 3.6236622746002953),
#         Cloudlet(0.081256, 149.824589, 243.16971522421468, 4.009965930243791),
#         Cloudlet(0.237547, 141.050771, 277.01199985466394, 4.671274901135505),
#         Cloudlet(0.138457, 139.508608, 271.25359518569496, 3.9828754698861477),
#         Cloudlet(0.088451, 133.618232, 245.98393640211285, 3.81448563152322),
#         Cloudlet(0.266167, 156.087665, 214.0395006818089, 5.657246768827748),
#         Cloudlet(0.130581, 158.033508, 251.24327206708733, 5.252957834065088),
#         Cloudlet(0.099247, 211.409329, 197.81288865451026, 4.240369159034978),  # 16
#         Cloudlet(0.124647, 259.696868, 245.59672377663492, 7.850605743087694),
#         Cloudlet(0.076976, 186.666789, 277.3108057619953, 2.440325446644967),  # 18
#     ]
#     ac = ACScheduler(lets, nodes, times=150)
#     res = ac.scheduler_main()
#     i = 0
#     for _ in ac.best_topo:
#         print("任务:", i, " 放置到机器", ac.best_topo[i], "上执行")
#         i += 1
#     plt.plot(range(len(res)), res)
#     plt.xlabel("迭代次数")
#     plt.ylabel("适应度")
#     plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
#     plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
#     plt.title("蚁群算法求解云任务调度负载均衡问题")
#     # # plt.savefig('imgr2/ACScheduler-1.1_0.9-popu100-iter200.png', dpi=300,
#     # #             format='png')  # bbox_inches="tight"解决X轴时间两个字不被保存的问题
#     plt.show()
