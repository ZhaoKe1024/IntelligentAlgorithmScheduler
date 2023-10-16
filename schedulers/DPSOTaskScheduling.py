import numpy as np
from matplotlib import pyplot as plt

from utils.Entities import Cloudlet, VM, calculate_fitness
from utils.AlgorithmEntities import DParticle


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
        result = self.exec()
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


if __name__ == '__main__':
    # 测试数据
    nodes = [
        VM(0, 0.762, 2, 920, 2223, 400, 2000, 5.9, 30),
        VM(1, 0.762, 2, 1200, 2223, 1000, 2000, 6, 30),
        VM(2, 0.762, 2, 850, 2223, 800, 2000, 5.8, 30),
        VM(3, 0.762, 2, 1200, 2223, 900, 2000, 5.9, 30),  # 4
    ]
    lets = [
        Cloudlet(0.078400, 60.689797, 228.9767525518272, 2.712677828249846),
        Cloudlet(0.065683, 185.848012, 187.97925460500625, 5.1178778788024),
        Cloudlet(0.050440, 96.030497, 206.77315938787453, 4.264445831060432),
        Cloudlet(0.104019, 131.428883, 218.78608382384854, 2.209277743955084),  # 4
        Cloudlet(0.022355, 192.582491, 231.9710696727387, 3.26584657336946),
        Cloudlet(0.232862, 226.085299, 233.03395445541793, 4.289629843497603),
        Cloudlet(0.194654, 77.503350, 190.41556439297744, 4.626189837323374),
        Cloudlet(0.148194, 241.349622, 264.54311244786555, 4.095493414214854),  # 8
        Cloudlet(0.146926, 199.978750, 248.2824412513349, 3.6236622746002953),
        Cloudlet(0.081256, 149.824589, 243.16971522421468, 4.009965930243791),
        Cloudlet(0.237547, 141.050771, 277.01199985466394, 4.671274901135505),
        Cloudlet(0.138457, 139.508608, 271.25359518569496, 3.9828754698861477),
        Cloudlet(0.088451, 133.618232, 245.98393640211285, 3.81448563152322),
        Cloudlet(0.266167, 156.087665, 214.0395006818089, 5.657246768827748),
        Cloudlet(0.130581, 158.033508, 251.24327206708733, 5.252957834065088),
        Cloudlet(0.099247, 211.409329, 197.81288865451026, 4.240369159034978),  # 16
        Cloudlet(0.124647, 259.696868, 245.59672377663492, 7.850605743087694),
        Cloudlet(0.076976, 186.666789, 277.3108057619953, 2.440325446644967),  # 18
    ]
    bpso = DPSO(lets, nodes, times=150)
    bpso.schedule()
