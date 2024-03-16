import numpy as np
from matplotlib import pyplot as plt

from eautils.Entities import calculate_fitness
from eautils.AlgorithmEntities import DParticle


class DPSO:
    def __init__(self, cloudlets, vms, population_number=100, times=500, w=0.95, c1=2, c2=2):
        self.cloudlets = cloudlets
        self.vms = vms
        self.population_number = population_number  # 种群数量
        self.times = times  # 遗传代数
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.vmax=5

        self.cloudlet_num = len(cloudlets)  # 任务数量也就是粒子长度
        self.machine_number = len(vms)  # 机器数量
        # self.chromosome_size = 0  # 染色体长度 int

        self.particles = set()  # Set[Gene]类型
        self.gbest = None

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
        E1 = np.array([np.random.random() for _ in range(self.cloudlet_num)])  # [0.1, 0.2, 0.002, 0.4, ...]
        E2 = np.array([np.random.random() for _ in range(self.cloudlet_num)])
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

    # 离散粒子群算法
    def execute(self):
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
            results.append(global_best_score)
            if t % 20 == 0:
                print("DPSO iter: ", t, " / ", self.times, ", 适应度: ", global_best_score)
        # return gb.solution
        return results

    # 输出结果
    def schedule(self):
        result = self.execute()
        # print("exec结果result：")
        # print(result)  # 打印出来是个引用，因为result是GeneEvaluation
        i = 0
        for _ in self.gbest.solution:
            print("任务:", i, " 放置到机器", self.vms[self.gbest.solution[i]].id+1, "上执行")
            i += 1
        # print(gb.solution)
        plt.plot(range(self.times), result)
        # plt.ylim(2.55 ,2.72)
        # plt.savefig('BPSOScheduler-popu100-iter150-w095-cg2-cl2.png', dpi=300, format='png')  # bbox_inches="tight"解决X轴时间两个字不被保存的问题
        plt.show()
