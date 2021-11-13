# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2021-03-15 0:57
# 参考自TS解决TSP问题：https://blog.csdn.net/qq_37325947/article/details/86727571

import numpy as np
import matplotlib.pyplot as plt
from utils.Entities import Cloudlet, VM


class TSScheduler:
    def __init__(self, cloudlets, vms, times=200):
        self.cloudlets = cloudlets
        self.vms = vms
        self.cloudlet_num = len(cloudlets)  # 任务数量也就是粒子长度
        self.machine_number = len(vms)  # 机器数量

        self.times = times  # 迭代代数

        self.TABLE_LEN = 8  # 禁忌长度
        self.SPE = 5  # 特赦值
        # 禁忌表
        self.tabu = [[0] * self.cloudlet_num for _ in range(self.cloudlet_num)]

        self.now_way = np.random.randint(0, self.machine_number, self.cloudlet_num)
        self.best_way = [0]*self.cloudlet_num
        self.best = 0  # 最优解
        self.p = []

    # 评价函数（和算法无关）
    def evaluate_particle(self, p) -> int:
        cpu_util = np.zeros(self.machine_number)
        mem_util = np.zeros(self.machine_number)
        for i in range(len(self.vms)):
            cpu_util[i] = self.vms[i].cpu_supply
            mem_util[i] = self.vms[i].mem_supply

        for i in range(self.cloudlet_num):
            cpu_util[p[i]] += self.cloudlets[i].cpu_demand
            mem_util[p[i]] += self.cloudlets[i].mem_demand

        for i in range(self.machine_number):
            if cpu_util[i] > self.vms[i].cpu_velocity:
                return 100
            if mem_util[i] > self.vms[i].mem_capacity:
                return 100

        for i in range(self.machine_number):
            cpu_util[i] /= self.vms[i].cpu_velocity
            mem_util[i] /= self.vms[i].mem_capacity

        return np.std(cpu_util, ddof=1) + np.std(mem_util, ddof=1)

    # 适应度函数（和算法无关）
    def calculate_fitness(self, p) -> float:
        return 1 / self.evaluate_particle(p)

    def cop(self, a, b):  # 把b数组的值赋值a数组
        for i in range(self.cloudlet_num):
            a[i] = b[i]

    # 主要流程
    def taboo_search_main(self):
        now = self.calculate_fitness(self.now_way)
        self.cop(self.best_way, self.now_way)
        self.best = now
        results = np.zeros(self.times)
        for t in range(self.times):
            temp = [0]*self.cloudlet_num
            a = 0
            b = 0
            ob_way = [0]*self.cloudlet_num
            self.cop(ob_way, self.now_way)
            ob_value = self.calculate_fitness(self.now_way)
            for i in range(self.cloudlet_num):
                for j in range(self.cloudlet_num):
                    if i + j >= self.cloudlet_num:
                        break
                    if i == j:
                        continue
                    self.cop(temp, self.now_way)
                    temp[i], temp[i + j] = temp[i + j], temp[i]
                    value = self.calculate_fitness(temp)
                    if value > self.best and self.tabu[i][i + j] < self.SPE:
                        self.cop(self.best_way, temp)
                        self.best = value
                        a = i
                        b = i + j
                    elif self.tabu[i][i + j] == 0 and value > ob_value:
                        self.cop(ob_way, temp)
                        ob_value = value
                        a = i
                        b = i + j
            self.cop(self.now_way, ob_way)
            for i in range(self.cloudlet_num):
                for j in range(self.cloudlet_num):
                    if self.tabu[i][j] > 0:
                        self.tabu[i][j] -= 1
            self.tabu[a][b] = self.TABLE_LEN
            results[t] = self.best

            if t % 10 == 0:
                print('iter: ', t, '/', self.times, "，适应度：", self.best)

        # print(results)
        print("最佳适应度：", self.best)
        # return expect_best
        # plt.plot(np.arange(self.times), results)
        # plt.show()
        return results


# if __name__ == '__main__':
#     # 测试数据
#     vms = [VM(0, 0.862, 2, 950, 1719),
#            VM(1, 0.962, 2, 950, 1719),
#            VM(2, 1.062, 2, 1500, 1719)]
#     lets = [Cloudlet(0.15, 50), Cloudlet(0.05, 100), Cloudlet(0.2, 60),
#             Cloudlet(0.01, 70), Cloudlet(0.04, 80), Cloudlet(0.07, 20),
#             Cloudlet(0.14, 150), Cloudlet(0.15, 200), Cloudlet(0.03, 40), Cloudlet(0.06, 90)]
#     ts = TSScheduler(lets, vms, times=150)
#     res = ts.taboo_search_main()
#     i = 0
#     for _ in ts.best_way:
#         print("任务:", i, " 放置到机器", vms[ts.best_way[i]].id, "上执行")
#         i += 1
