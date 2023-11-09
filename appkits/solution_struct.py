#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/9 10:07
# @Author: ZhaoKe
# @File : solution_struct.py
# @Software: PyCharm
import numpy as np


class SimpleSolution(object):
    """
    @solution: the solution vector, 1dim
    fitness: a scalar value
    """
    def __init__(self, solution=None, fitness=None):
        self.solution = solution
        self.fitness = fitness

    def set_solution(self, solution):
        self.solution = solution

    def set_fitness(self, fitness):
        self.fitness = fitness

    def __str__(self):
        if self.solution is not None:
            return '[' + ", ".join([str(s) for s in self.solution]) + ']'
        else:
            return "[]"


class SimpleSolutionGenerator(object):
    def __init__(self, task_num, machine_num):
        self.task_num = task_num
        self.machine_num = machine_num

    def initialize_solution(self):
        return SimpleSolution(solution=np.random.randint(0, self.machine_num, (self.task_num)), fitness=None)


if __name__ == '__main__':
    so_generator = SimpleSolutionGenerator(10, 3)
    so = so_generator.initialize_solution()
    print(so)
