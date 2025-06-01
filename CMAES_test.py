#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2025/3/30 10:18
# @Author: ZhaoKe
# @File : CMAES_test.py
# @Software: PyCharm
import cma
import numpy as np


# 定义目标函数（香蕉函数）
def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


# 使用 CMA-ES 求解
if __name__ == "__main__":
    # 初始参数
    x0 = [0., 0.]
    sigma0 = 0.5
    options = {
        'bounds': [-5., 5.],
        # 'verbose': -9,
        'maxfevals': 1000
    }
    cmaes = cma.CMAEvolutionStrategy(x0, sigma0, options)
    ind = 0
    while not cmaes.stop():
        solutions = cmaes.ask()  # generate solutions
        fitness = [rosenbrock(x) for x in solutions]
        cmaes.tell(solutions, fitness)  # update parameters
        cmaes.logger.add(cmaes)  # logging
        print(ind)
        ind += 1

        best_solution = cmaes.result.xbest
        best_fitness = cmaes.result.fbest
        print("\nOptimization Results:")
        print("Optimal:".format(best_solution))
        print("Minimum:".format(best_fitness))
