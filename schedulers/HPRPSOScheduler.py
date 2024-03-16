# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2021-03-17 11:06
# binary particle
import numpy as np
from matplotlib import pyplot as plt

from eautils.Entities import calculate_fitness
from eautils.dataExamples import get_data_r2n3c12
from eautils.AlgorithmEntities import DParticle


class HPRPSO:
    def __init__(self, cloudlets, vms, population_number=1000, times=500, w=0.95, c1=2, c2=2):
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

        self.particles = list()  # Set[Gene]类型
        self.gbest = None

    def init_population(self):
        for _ in range(self.population_number):
            size = self.cloudlet_num
            p = DParticle(size)
            p.solution = np.random.randint(0, self.machine_number, size=self.cloudlet_num)
            p.velocity = -5 + 10 * np.random.rand(self.cloudlet_num)
            p.fitness = calculate_fitness(p.solution, self.cloudlets, self.vms)
            self.particles.append(p)

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
        E1 = np.array([np.random.random() for _ in range(self.cloudlet_num)])  # [0.1, 0.2, 0.002, 0.4, ...]
        E2 = np.array([np.random.random() for _ in range(self.cloudlet_num)])
        v1 = np.array(gb.solution) - np.array(p.solution)
        v2 = np.array(lb.solution) - np.array(p.solution)
        velocity = self.c1 * E1 * v1 + self.c2 * E2 * v2
        velocity = np.clip(velocity, -self.vmax, self.vmax)
        p.velocity = p.velocity * self.w + velocity

    def update_position_copy1(self, p: DParticle, b: DParticle, flag: int):
        # 复制0.8即可
        if flag == 1:
            le = int(self.cloudlet_num * 0.7)
            start = np.random.randint(0, self.cloudlet_num)
            for i in range(start, start+le):
                p.solution[i % self.cloudlet_num] = b.solution[i % self.cloudlet_num]
        if flag == 2:
            le = int(self.cloudlet_num * 0.4)
            start = np.random.randint(0, self.cloudlet_num)
            for i in range(start, start+le):
                p.solution[i % self.cloudlet_num] = b.solution[i % self.cloudlet_num]

    def update_position(self, p: DParticle):
        p.solution = np.abs(p.solution + p.velocity)
        for t in range(len(self.cloudlets)):
            if p.solution[t] > self.machine_number:
                p.solution[t] = np.ceil(p.solution[t]) % self.machine_number
        p.solution = list(map(int, p.solution))

    # 离散粒子群算法
    def exec(self):
        # 初始化基因，基因包含染色体和染色体的适应度
        self.init_population()
        self.gbest = self.find_best()
        lb = self.particles.pop()
        best_score = calculate_fitness(lb.solution, self.cloudlets, self.vms)
        global_best_score = calculate_fitness(self.gbest.solution, self.cloudlets, self.vms)
        results = []
        # 迭代过程 仅包含速度和位置的更新
        for t in range(self.times):
            # 升序排序
            self.particles.sort(key=lambda x: x.fitness)
            ind = 0
            for p in self.particles:
                if p is self.gbest:
                    continue
                if p is lb:
                    continue
                self.update_velocity(p, lb, self.gbest)
                self.update_position(p)
                if ind > self.population_number / 3 or ind < 2 / 3 * self.population_number:
                    self.update_position_copy1(p, lb, 2)
                elif ind > 2/3 * self.population_number:
                    self.update_position_copy1(p, self.gbest, 1)
                ind += 1
                score = calculate_fitness(p.solution, self.cloudlets, self.vms)
                if score > best_score:
                    best_score = score
                    lb = p
                if score > global_best_score:
                    global_best_score = score
                    self.gbest = p
            results.append(global_best_score)
            if t % 20 == 0:
                print("HPRPSO iter: ", t, " / ", self.times, ", 适应度: ", global_best_score)
        # print(gb.solution)
        return results

    # 输出结果
    def schedule(self):
        # bpso = BPSO(self.cloudlets, self.vms, times=300)
        result = self.exec()
        # print("exec结果result：")
        # print(result)  # 打印出来是个引用，因为result是GeneEvaluation
        i = 0
        for _ in self.gbest.solution:
            print("任务:", i, " 放置到机器", self.vms[self.gbest.solution[i]].id + 1, "上执行")
            i += 1
        plt.plot(range(self.times), result)
        # plt.savefig('imgr2/BPOScheduler-0.95_2_2--vmax5-popu100-iter200-w095-cg2-cl2.png', dpi=300,
        #             format='png')  # bbox_inches="tight"解决X轴时间两个字不被保存的问题
        plt.show()


if __name__ == '__main__':
    # 测试数据
    data = get_data_r2n3c12(1)
    nodes = data["nodes"]
    lets = data["cloudlets"]
    # lets.sort(key=lambda x: x.mem_demand)
    # for let in lets:
    #     print(let.mem_demand)
    nspso = HPRPSO(lets, nodes, population_number=100, times=150)
    nspso.schedule()
