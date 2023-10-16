# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2021-03-21 10:43
import numpy as np
from utils.Entities import Cloudlet, VM, calculate_fitness
from matplotlib import pyplot as plt


class Gene:
    def __init__(self):
        """
        @solution: the solution vector, 1dim
        fitness: a scalar value
        """
        self.solution = None
        self.fitness = None


class GAScheduler:
    def __init__(self, cloudlets, vms, population_number=100, times=500):
        self.cloudlets = cloudlets
        self.vms = vms
        self.cloudlet_num = len(cloudlets)  # 任务数量也就是粒子长度
        self.machine_number = len(vms)  # 机器数量

        self.population_number = population_number  # 种群数量
        self.times = times  # 迭代代数

        # 变异和交叉概率
        self.mp = 0.1
        self.cp = 0.9

        self.best_gene = None

        self.genes = list()

    # 初始化群体
    def init_population(self):
        for i in range(self.population_number):
            gene = Gene()
            gene.solution = np.random.randint(0, self.machine_number, self.cloudlet_num)
            gene.fitness = calculate_fitness(gene.solution, self.cloudlets, self.vms)
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
                new_gene = Gene()
                new_gene.solution = np.hstack((g1.solution[0:point], g2.solution[point:]))
                new_gene.fitness = calculate_fitness(new_gene.solution, self.cloudlets, self.vms)
                self.genes[i] = new_gene

    # 随机单点变异
    def mutation(self):
        for i in range(self.population_number):
            point = np.random.randint(0, self.cloudlet_num)
            if np.random.rand() < self.mp:
                self.genes[i].solution[point] = np.random.randint(0, self.machine_number)
                self.genes[i].fitness = calculate_fitness(self.genes[i].solution, self.cloudlets, self.vms)

    # 选择群体最优个体
    def select_best(self) -> Gene:
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
            best_g = self.select_best()
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
                print("GA iter: ", t, "/", self.times, "适应度: ", local_gene.fitness)
            if local_gene.fitness > self.best_gene.fitness:
                self.best_gene = local_gene
            # print("最优解：", self.genes[ind].solution)
        # return results

        # plt.plot(range(self.times), results)  # 正常应该是2.7左右
        # plt.savefig('imgr2/BPOScheduler-0.95_2_2--vmax5-popu100-iter200-w095-cg2-cl2.png', dpi=300,
        #             format='png')  # bbox_inches="tight"解决X轴时间两个字不被保存的问题
        # plt.show()
        return results


if __name__ == '__main__':
    nodes = [
        VM(0, 0.762, 2, 920, 2223, 400, 2000),
        VM(1, 0.762, 2, 1200, 2223, 1000, 2000),
        VM(2, 0.762, 2, 850, 2223, 800, 2000),
        VM(3, 0.762, 2, 1200, 2223, 900, 2000),  # 4
    ]
    lets = [
        Cloudlet(0.078400, 60.689797, 228.9767518),
        Cloudlet(0.065683, 185.848012, 187.979460),
        Cloudlet(0.050440, 96.030497, 206.7731532),
        Cloudlet(0.104019, 131.428883, 218.7860084),  # 4
        Cloudlet(0.022355, 192.582491, 231.97106946),
        Cloudlet(0.232862, 226.085299, 233.033903),
        Cloudlet(0.194654, 77.503350, 190.41556474),
        Cloudlet(0.148194, 241.349622, 264.54314854),  # 8
        Cloudlet(0.146926, 199.978750, 248.28244),
        Cloudlet(0.081256, 149.824589, 243.1697241),
        Cloudlet(0.237547, 141.050771, 277.01199505),
        Cloudlet(0.138457, 139.508608, 271.253561477),
        Cloudlet(0.088451, 133.618232, 245.98393322),
        Cloudlet(0.266167, 156.087665, 214.0397748),
        Cloudlet(0.130581, 158.033508, 251.243065088),
        Cloudlet(0.099247, 211.409329, 197.81288978),  # 16
        Cloudlet(0.124647, 259.696868, 245.596727694),
        Cloudlet(0.076976, 186.666789, 277.310805967),  # 18
    ]
    ga = GAScheduler(lets, nodes, times=200)
    res = ga.execute()
    print("最高适应度:", ga.best_gene.fitness)
    # i = 0
    # for _ in ga.best_gene.solution:
    #     print("任务:", i, " 放置到机器", ga.vms[ga.best_gene.solution[i]].id, "上执行")
    #     i += 1
    plt.plot(range(ga.times), res)
    plt.show()
