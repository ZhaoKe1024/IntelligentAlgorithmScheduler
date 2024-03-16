# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2021-03-20 17:50
import numpy as np
from eautils.Entities import Cloudlet, VM, calculate_fitness
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

    def execute(self):
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
