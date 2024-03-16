# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2021-03-17 11:06
# binary particle
import numpy as np
from eautils.Entities import Cloudlet, VM
from matplotlib import pyplot as plt


# 粒子代表问题的一个解，一个解决方案，由01标识，表示粒子的位置向量，另外还有速度向量
# 使用取整求余的方法解决粒子群的离散应用问题，因此这里速度还是原来的含义，而位置的含义变为任务在哪个计算节点上执行
class Bparticle:
    def __init__(self, cloudlet_num: int, fitness: float = 0):
        self.solution = np.array([0 for _ in range(cloudlet_num)], dtype=int)  # 获得一个长度为size的01随机向量
        self.velocity = np.array([0 for _ in range(cloudlet_num)], dtype=float)  # 获得一个长度为size的零向量
        self.fitness = fitness

    def __str__(self):
        result = [str(e) for e in self.solution]  # 例如['0', '1', '0', '1', '1', '1', '0', '0']
        return '[' + ', '.join(result) + ']'  # 例如'[0,1,0,1,1,1,0,0]'


class NSPSO:
    def __init__(self, cloudlets, vms, population_number=500, times=500, w=0.95, c1=2, c2=2):
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

    # 评价函数
    def evaluate_particle(self, p: Bparticle) -> int:
        cpu_util = np.zeros(self.machine_number)
        mem_util = np.zeros(self.machine_number)
        for i in range(len(self.vms)):
            cpu_util[i] = self.vms[i].cpu_supply
            mem_util[i] = self.vms[i].mem_supply

        for i in range(self.cloudlet_num):
            cpu_util[p.solution[i]] += self.cloudlets[i].cpu_demand
            mem_util[p.solution[i]] += self.cloudlets[i].mem_demand

        for i in range(self.machine_number):
            if cpu_util[i] > self.vms[i].cpu_velocity:
                return 100
            if mem_util[i] > self.vms[i].mem_capacity:
                return 100

        for i in range(self.machine_number):
            cpu_util[i] /= self.vms[i].cpu_velocity
            mem_util[i] /= self.vms[i].mem_capacity

        return np.std(cpu_util, ddof=1) + np.std(mem_util, ddof=1)

    def calculate_fitness(self, p: Bparticle) -> float:
        return 1 / self.evaluate_particle(p)

    def init_population(self):
        for _ in range(self.population_number):
            size = self.cloudlet_num
            p = Bparticle(size)
            p.solution = np.random.randint(0, self.machine_number, size=self.cloudlet_num)
            p.velocity = -5 + 10 * np.random.rand(self.cloudlet_num)
            p.fitness = self.calculate_fitness(p)
            self.particles.append(p)

    def init_new_population(self):
        particles = list()
        for _ in range(self.population_number):
            size = self.cloudlet_num
            p = Bparticle(size)
            p.solution = np.random.randint(0, self.machine_number, size=self.cloudlet_num)
            p.velocity = -5 + 10 * np.random.rand(self.cloudlet_num)
            p.fitness = self.calculate_fitness(p)
            particles.append(p)
        particles.sort(key=lambda x: x.fitness)
        return particles

    def find_best(self):
        best_score = 0
        best_particle = None
        for p in self.particles:
            score = self.calculate_fitness(p)
            if score > best_score:
                best_score = score
                best_particle = p
        return best_particle

    def find_worse(self):
        best_score = 0
        best_particle = None
        for p in self.particles:
            score = self.calculate_fitness(p)
            if score < best_score:
                best_score = score
                best_particle = p
        return best_particle

    def update_velocity(self, p: Bparticle, lb: Bparticle, gb: Bparticle):
        E1 = np.array([np.random.random() for _ in range(self.cloudlet_num)])  # [0.1, 0.2, 0.002, 0.4, ...]
        E2 = np.array([np.random.random() for _ in range(self.cloudlet_num)])
        v1 = np.array(gb.solution) - np.array(p.solution)
        v2 = np.array(lb.solution) - np.array(p.solution)
        velocity = self.c1 * E1 * v1 + self.c2 * E2 * v2
        velocity = np.clip(velocity, -self.vmax, self.vmax)
        p.velocity = p.velocity * self.w + velocity + np.random.randint(0, 8)

    def update_position_copy1(self, p: Bparticle, b: Bparticle, flag: int):
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

    def update_position(self, p: Bparticle):
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
        best_score = self.calculate_fitness(lb)
        global_best_score = self.calculate_fitness(self.gbest)
        results = []
        # 迭代过程 仅包含速度和位置的更新
        for t in range(self.times):
            # 升序排序
            self.particles.sort(key=lambda x: x.fitness)
            new_particles = self.init_new_population()
            for i in range(100):
                self.particles[-1-i] = new_particles[i]
            # p = []

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
                else:
                    self.update_position_copy1(p, self.gbest, 1)
                ind += 1
                score = self.calculate_fitness(p)
                if score > best_score:
                    best_score = score
                    lb = p
                if score > global_best_score:
                    global_best_score = score
                    self.gbest = p
            results.append(global_best_score)
            # if t % 10 == 0:
            print("NSPSO iter: ", t, " / ", self.times, ", 适应度: ", global_best_score)
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
        Cloudlet(0.196168, 176.903110),
        Cloudlet(0.307672, 102.491285),
        Cloudlet(0.044395, 248.902957),
        Cloudlet(0.224083, 36.589774),
        Cloudlet(0.177338, 147.456287),
        Cloudlet(0.037426, 137.866169),
        Cloudlet(0.112618, 176.490872),
        Cloudlet(0.179390, 176.053534),
        Cloudlet(0.457740, 94.916472),
        Cloudlet(0.388151, 152.429698),
        Cloudlet(0.033796, 143.141112),
        Cloudlet(0.410989, 197.744171),  # 12
        Cloudlet(0.212318, 229.817538),
        Cloudlet(0.144493, 230.920339),
        Cloudlet(0.211401, 95.001009),
        Cloudlet(0.132285, 159.829442),
        Cloudlet(0.139237, 70.856739),
        Cloudlet(0.278064, 77.788423),
        Cloudlet(0.271126, 154.028134),
        Cloudlet(0.271827, 260.086359),
        Cloudlet(0.207681, 101.475979),
        Cloudlet(0.046045, 180.085124),
        Cloudlet(0.211615, 138.958982),
        Cloudlet(0.290154, 231.477185),  # 24
        Cloudlet(0.191973, 79.471904),
        Cloudlet(0.238924, 156.742957),
        Cloudlet(0.212445, 192.564833),
        Cloudlet(0.123814, 230.323520),
        Cloudlet(0.175196, 260.884240),
        Cloudlet(0.082192, 160.419990),
        Cloudlet(0.226339, 51.740979),
        Cloudlet(0.051243, 103.361152),
        Cloudlet(0.057970, 81.365240),
        Cloudlet(0.080281, 316.428299),
        Cloudlet(0.103359, 112.089798),
        Cloudlet(0.273650, 206.036697),  # 36
    ]
    # lets.sort(key=lambda x: x.mem_demand)
    # for let in lets:
    #     print(let.mem_demand)
    nspso = NSPSO(lets, nodes, population_number=500, times=150)
    nspso.schedule()

