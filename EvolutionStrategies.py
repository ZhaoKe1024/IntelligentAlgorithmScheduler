#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2025/3/23 19:39
# @Author: ZhaoKe
# @File : EvolutionStrategies.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt


# 目标函数（香蕉函数/Rosenbrock函数）
def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


# 高斯进化策略实现
class GaussianEvolutionStrategy:
    def __init__(self,
                 mu_init,
                 sigma_init,
                 population_size=50,
                 elite_size=10,
                 max_iter=1000,
                 tolerance=1e-6):
        self.mu = np.array(mu_init, dtype=np.float64)
        self.sigma = sigma_init
        self.population_size = population_size
        self.elite_size = elite_size
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.history = []

    def optimize(self):
        best_fitness = np.inf
        convergence_counter = 0

        for generation in range(self.max_iter):
            # 生成种群
            population = self.mu + self.sigma * np.random.randn(
                self.population_size,
                len(self.mu)
            )

            # 评估适应度
            fitness = np.array([rosenbrock(x) for x in population])

            # 选择精英个体
            elite_indices = np.argsort(fitness)[:self.elite_size]
            elites = population[elite_indices]

            # 更新参数
            new_mu = np.mean(elites, axis=0)
            new_sigma = np.mean(np.std(elites, axis=0))  # 各维度标准差的均值

            # 收敛检测
            current_best = np.min(fitness)
            self.history.append(current_best)

            if np.abs(best_fitness - current_best) < self.tolerance:
                convergence_counter += 1
                if convergence_counter >= 5:  # 连续5代变化小于阈值
                    break
            else:
                convergence_counter = 0

            best_fitness = current_best
            self.mu = new_mu
            self.sigma = max(new_sigma, 1e-3)  # 防止sigma过小

            # 打印进度
            if generation % 10 == 0:
                print(f"Gen {generation}: μ={self.mu.round(4)}, σ={self.sigma:.4f}, "
                      f"Best Fitness={current_best:.6f}")

        return self.mu, self.history


# 运行优化
def run_GES():
    # 初始化参数
    optimizer = GaussianEvolutionStrategy(
        mu_init=[-5.0, 5.0],  # 初始均值
        sigma_init=2.0,  # 初始标准差
        population_size=100,
        elite_size=10,
        max_iter=200
    )

    # 执行优化
    optimal_solution, history = optimizer.optimize()

    # 输出结果
    print("\nOptimization Results:")
    print(f"最优解: {optimal_solution}")
    print(f"最小函数值: {rosenbrock(optimal_solution):.6f}")

    # 可视化收敛过程
    plt.plot(history, label="Best Fitness")
    plt.yscale("log")
    plt.xlabel("Generation")
    plt.ylabel("Function Value")
    plt.title("Convergence of Gaussian Evolution Strategy")
    plt.legend()
    plt.show()


class CMA_ES:
    def __init__(self,
                 x0,
                 sigma0=0.5,
                 population_size=20,
                 elite_ratio=0.5,
                 max_iter=500,
                 tol=1e-6):
        """
        协方差矩阵的更新通常包括两个部分：基于精英个体的分布和演化路径的累积。例如，使用权重系数来加权精英个体的贡献，并更新协方差矩阵的路径。
        """
        self.n_dim = len(x0)
        self.population_size = population_size
        self.elite_size = int(population_size * elite_ratio)
        self.sigma = sigma0
        self.tol = tol

        # 初始化参数 [[1]][[9]]
        self.mu = np.array(x0, dtype=np.float64)
        self.C = np.eye(self.n_dim)  # 协方差矩阵
        self.pc = np.zeros(self.n_dim)  # 演化路径
        self.cc = 0.5  # 演化路径学习率
        self.ccov = 0.5  # 协方差学习率

        self.history = []
        self.max_iter = max_iter

    def get_random_population(self):
        # 生成候选解 [[9]]
        return [self.mu + self.sigma * np.random.multivariate_normal(np.zeros(self.n_dim), self.C)
                for _ in range(self.population_size)]

    def update(self, population, fitness):
        # 选择精英个体 [[1]]
        elite_indices = np.argsort(fitness)[:self.elite_size]
        elites = population[elite_indices]
        # 更新均值 [[10]]
        old_mu = self.mu.copy()
        self.mu = np.mean(elites, axis=0)
        # 更新演化路径 [[5]]
        self.pc = (1 - self.cc) * self.pc + np.sqrt(self.cc * (2 - self.cc)) * (self.mu - old_mu) / self.sigma
        # 更新协方差矩阵 [[1]][[5]]
        self.C = (1 - self.ccov) * self.C + self.ccov * np.outer(self.pc, self.pc)
        # 更新步长 [[9]]
        self.sigma *= np.exp((np.linalg.norm(self.pc) / self.n_dim - 1) * 0.3)

    def optimize(self):
        best_fitness = np.inf
        for gen in range(self.max_iter):
            population = np.array(self.get_random_population())
            fitness = np.array([rosenbrock(x) for x in population])

            # 记录最佳解
            current_best = np.min(fitness)
            self.history.append(current_best)

            # 收敛检测
            if np.abs(best_fitness - current_best) < self.tol:
                print(f"Converged at generation {gen}")
                break
            best_fitness = current_best

            # 更新参数
            self.update(population, fitness)

            # 显示进度
            if gen % 10 == 0:
                print(f"Gen {gen}: μ={self.mu.round(4)}, σ={self.sigma:.4f}, "
                      f"Best={current_best:.6f}")

        return self.mu, self.history


if __name__ == "__main__":
    optimizer = CMA_ES(x0=[-5.0, 5.0], sigma0=2.0, population_size=50)
    solution, history = optimizer.optimize()
    print("\nOptimization Results:")
    print(f"最优解: {solution}")
    print(f"最小函数值: {rosenbrock(solution):.6f}")

    # 可视化收敛过程
    plt.plot(history, label="CMA-ES")
    plt.yscale("log")
    plt.xlabel("Generation")
    plt.ylabel("Function Value")
    plt.title("CMA-ES Optimization on Rosenbrock")
    plt.legend()
    plt.show()
