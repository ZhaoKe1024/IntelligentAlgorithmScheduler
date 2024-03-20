# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2021-03-21 10:43
from copy import deepcopy

import numpy as np
from eautils.Entities import calculate_fitness
from eautils.solution_struct import SimpleSolution, SimpleSolutionGenerator


class GAScheduler:
    def __init__(self, cloudlets, vms, population_number=100, times=800):
        self.cloudlets = cloudlets
        self.vms = vms
        self.cloudlet_num = len(cloudlets)  # 任务数量也就是粒子长度
        self.machine_number = len(vms)  # 机器数量

        self.population_number = population_number  # 种群数量
        self.times = times  # 迭代代数

        # 变异和交叉概率
        self.mp = 0.15
        self.cp = 0.85

        self.best_gene = None

        self.genes = list()
        self.so_generator = SimpleSolutionGenerator(self.cloudlet_num, self.machine_number)

    # 初始化群体
    def init_population(self):
        for i in range(self.population_number):
            gene = self.so_generator.initialize_solution()
            gene.set_fitness(calculate_fitness(gene.solution, self.cloudlets, self.vms))
            self.genes.append(gene)

    # 轮盘赌选取一个
    # 然后随机选取另一个
    def select(self):
        pa = np.zeros(self.population_number)
        ps = 0
        for g in self.genes:
            ps += g.fitness
        for i in range(self.population_number):
            pa[i] = self.genes[i].fitness / ps
        pa = np.cumsum(pa)
        g1 = None
        g2 = None
        for i in range(self.population_number):
            for j in range(self.population_number):
                r = np.random.rand()
                if pa[j] > r:
                    g2 = self.genes[j]
                    break
            g1 = self.genes[np.random.randint(0, self.population_number)]
        return g1, g2

    # 随机交换片段
    def crossover(self, g1, g2):
        for i in range(self.population_number):
            if np.random.rand() < self.cp:
                point = np.random.randint(0, self.cloudlet_num)
                new_gene = SimpleSolution()
                new_gene.solution = np.hstack((g1.solution[0:point], g2.solution[point:]))
                new_gene.fitness = calculate_fitness(new_gene.solution, self.cloudlets, self.vms)
                self.genes[i] = new_gene

    # 随机单点变异
    def mutation(self):
        for i in range(self.population_number):
            if np.random.rand() < self.mp:
                point = np.random.randint(0, self.cloudlet_num)
                new_gene = deepcopy(self.genes[i])
                new_gene.solution[point] = np.random.randint(0, self.machine_number)
                new_gene.fitness = calculate_fitness(new_gene.solution, self.cloudlets, self.vms)
                self.genes[i] = new_gene

    # 选择群体最优个体
    def select_best(self) -> SimpleSolution:
        best = self.genes[0]
        for g in self.genes:
            if best.fitness < g.fitness:
                best = g
        return best

    # 选取群体最差个体
    def select_worst(self) -> int:
        worst = self.genes[0]
        ind = 0
        for i in range(self.population_number):
            if worst.fitness > self.genes[i].fitness:
                worst = self.genes[i]
                ind = i
        return ind

    # 精英保存策略
    def eselect(self, best_g):
        fit = self.genes[0].fitness
        for i in range(self.population_number):
            if fit < self.genes[i].fitness:
                fit = self.genes[i].fitness
        if fit < best_g.fitness:
            self.genes[self.select_worst()] = best_g

    # 主流程
    def execute(self):
        self.init_population()
        self.best_gene = self.genes[0]
        results = []
        for t in range(self.times):
            best_g = deepcopy(self.select_best())
            # 选择,
            g1, g2 = self.select()
            # 交叉
            self.crossover(g1, g2)
            # 变异
            self.mutation()
            # 精英保留
            self.eselect(best_g)

            local_gene = self.genes[0]
            for g in self.genes:
                if g.fitness > local_gene.fitness:
                    local_gene = g
            results.append(local_gene.fitness)
            if t % 20 == 0:
                print("GA iter: ", t, "/", self.times, "适应度 local: ", local_gene.fitness)
            if local_gene.fitness > self.best_gene.fitness:
                self.best_gene = local_gene
            if t % 20 == 0:
                print("GA iter: ", t, "/", self.times, "适应度 best: ", self.best_gene.fitness)
            # print("最优解：", self.genes[ind].solution)
        # return results

        # plt.plot(range(self.times), results)  # 正常应该是2.7左右
        # plt.savefig('imgr2/BPOScheduler-0.95_2_2--vmax5-popu100-iter200-w095-cg2-cl2.png', dpi=300,
        #             format='png')  # bbox_inches="tight"解决X轴时间两个字不被保存的问题
        # plt.show()
        return results
