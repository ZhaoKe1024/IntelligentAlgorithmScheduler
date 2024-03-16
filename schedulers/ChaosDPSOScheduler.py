# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2021-04-22 8:05
import numpy as np
from eautils.Entities import calculate_fitness
from eautils.functions import logistic_function
from matplotlib import pyplot as plt
from schedulers.DPSOTaskScheduling import DParticle


class ChaosDPSO:
    def __init__(self, cloudlets, vms, population_number=100, times=500, w=0.95, c1=2, c2=2):
        self.cloudlets = cloudlets
        self.vms = vms
        self.population_number = population_number  # 种群数量
        self.times = times  # 遗传代数
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.vmax = 5

        self.cloudlet_num = len(cloudlets)  # 任务数量也就是粒子长度
        self.machine_number = len(vms)  # 机器数量
        # self.chromosome_size = 0  # 染色体长度 int

        self.particles = set()  # Set[Gene]类型
        self.gbest = None

        self.sigma = 2
        self.chaosNum = 20

    def init_population(self):
        for _ in range(self.population_number):
            size = self.cloudlet_num
            p = DParticle(size)
            p.solution = np.random.randint(0, self.machine_number, size=self.cloudlet_num)
            p.velocity = -5 + 10 * np.random.rand(self.cloudlet_num)
            # print("chromosome:", g.chromosome)
            p.fitness = calculate_fitness(p.solution, self.cloudlets, self.vms)
            self.particles.add(p)

    def find_best(self):
        best_score = 0
        best_particle = None
        for p in self.particles:
            score = calculate_fitness(p.solution, self.cloudlets, self.vms)
            if score > best_score:
                best_score = score
                best_particle = p
        return best_particle

    def update_velocity(self, p: DParticle, lb: DParticle, gb: DParticle):
        E1 = np.array([np.random.random() for n in range(self.cloudlet_num)])  # [0.1, 0.2, 0.002, 0.4, ...]
        E2 = np.array([np.random.random() for n in range(self.cloudlet_num)])
        v1 = np.array(gb.solution) - np.array(p.solution)
        v2 = np.array(lb.solution) - np.array(p.solution)
        velocity = self.c1 * E1 * v1 + self.c2 * E2 * v2
        velocity = np.clip(velocity, -self.vmax, self.vmax)
        p.velocity = p.velocity * self.w + velocity

    def update_position(self, p: DParticle):
        p.solution = np.abs(p.solution + p.velocity)
        for t in range(len(self.cloudlets)):
            if p.solution[t] > self.machine_number:
                p.solution[t] = np.ceil(p.solution[t]) % self.machine_number
        p.solution = list(map(int, p.solution))

    def getSeries(self, v0):
        tempSet = [[0] * self.cloudlet_num] * 30
        tempSet[0] = [x / self.machine_number for x in v0]
        for i in range(1, 30):
            tempSet[i] = [x for x in logistic_function(tempSet[i - 1], 3.5)]
        for i in range(30):
            tempSet[i] = [int(self.machine_number * x) for x in tempSet[i]]
        # print(tempSet)
        tempFit = []
        for i in range(30):
            tempFit.append(calculate_fitness(tempSet[i], self.cloudlets, self.vms))
        best_score = 0
        best_solution = None
        for i in range(self.chaosNum):
            if tempFit[i] > best_score:
                best_score = tempFit[i]
                best_solution = tempSet[i]
        res = DParticle(self.cloudlet_num)
        res.velocity = self.gbest.velocity
        res.solution = best_solution
        res.fitness = best_score
        return res

    # 离散粒子群算法
    def exec(self, p):
        # 初始化基因，基因包含染色体和染色体的适应度
        self.init_population()
        self.gbest = self.find_best()
        lb = self.particles.pop()
        best_score = calculate_fitness(lb.solution, self.cloudlets, self.vms)
        global_best_score = calculate_fitness(self.gbest.solution, self.cloudlets, self.vms)
        results = []
        # 迭代过程 仅包含速度和位置的更新
        for t in range(self.times):
            for p in self.particles:
                if p is self.gbest:
                    continue
                if p is lb:
                    continue
                self.update_velocity(p, lb, self.gbest)
                self.update_position(p)
                score = calculate_fitness(p.solution, self.cloudlets, self.vms)
                if score > best_score:
                    best_score = score
                    lb = p
                if score > global_best_score:
                    global_best_score = score
                    self.gbest = p
            tempGbest = self.getSeries(self.gbest.solution)
            if tempGbest.fitness > global_best_score:
                global_best_score = tempGbest.fitness
                self.gbest = tempGbest
            results.append(global_best_score)
            if t % 20 == 0:
                print("CPSO iter: ", t, " / ", self.times, ", 适应度: ", global_best_score)
        # return gb.solution
        return results

    # 输出结果
    def schedule(self):
        result = self.exec()
        # print("exec结果result：")
        # print(result)  # 打印出来是个引用，因为result是GeneEvaluation
        i = 0
        for _ in self.gbest.solution:
            print("任务:", i, " 放置到机器", self.vms[self.gbest.solution[i]].id + 1, "上执行")
            i += 1
        # print(gb.solution)
        plt.plot(range(self.times), result)
        # plt.ylim(2.55 ,2.72)
        # plt.savefig('BPSOScheduler-popu100-iter150-w095-cg2-cl2.png', dpi=300, format='png')  # bbox_inches="tight"解决X轴时间两个字不被保存的问题
        plt.show()


# if __name__ == '__main__':
#     # 测试数据
#     nodes = [
#         VM(0, 0.762, 2, 920, 2223, 400, 2000),
#         VM(1, 0.762, 2, 1200, 2223, 1000, 2000),
#         VM(2, 0.762, 2, 850, 2223, 800, 2000),
#         VM(3, 0.762, 2, 1200, 2223, 900, 2000),  # 4
#     ]
#     lets = [
#         Cloudlet(0.078400, 60.689797, 228.9767549846),
#         Cloudlet(0.065683, 185.848012, 187.97925024),
#         Cloudlet(0.050440, 96.030497, 206.7730432),
#         Cloudlet(0.104019, 131.428883, 218.785084),  # 4
#         Cloudlet(0.022355, 192.582491, 231.971066946),
#         Cloudlet(0.232862, 226.085299, 233.033953),
#         Cloudlet(0.194654, 77.503350, 190.415564374),
#         Cloudlet(0.148194, 241.349622, 264.54314854),  # 8
#         Cloudlet(0.146926, 199.978750, 248.2824453),
#         Cloudlet(0.081256, 149.824589, 243.1697191),
#         Cloudlet(0.237547, 141.050771, 277.0119905),
#         Cloudlet(0.138457, 139.508608, 271.25358),
#         Cloudlet(0.088451, 133.618232, 245.98392322),
#         Cloudlet(0.266167, 156.087665, 214.03950067748),
#         Cloudlet(0.130581, 158.033508, 251.2432088),
#         Cloudlet(0.099247, 211.409329, 197.8128878),  # 16
#         Cloudlet(0.124647, 259.696868, 245.596727694),
#         Cloudlet(0.076976, 186.666789, 277.3108057617),  # 18
#     ]
#     bpso = ChaosDPSO(lets, nodes, times=150)
#     bpso.schedule()
