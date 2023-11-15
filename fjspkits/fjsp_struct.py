#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/15 20:53
# @Author: ZhaoKe
# @File : fjsp_struct.py
# @Software: PyCharm
import random

import numpy as np
import operator
from fjsp_utils import calculate_exetime_load


class Solution(object):
    def __init__(self, machines, job_num):
        self.__machines = machines
        self.__fitness = None
        self.job_num = job_num

    def set_machines(self, machines):
        self.__machines = machines

    def get_machines(self):
        return self.__machines

    def get_fitness(self):
        if self.__fitness is None:
            self.__fitness = calculate_exetime_load(self.__machines, self.job_num)
        return self.__fitness

    def set_fitness(self, value):
        self.__fitness = value


class SolutionSortedList(object):
    """为了优化进化计算中的自然选择算子，用排序列表来组织种群是最好的"""
    def __init__(self, desc=False):
        self.solutions = []
        self.desc = desc

    def update_fitness(self):
        """初始化适应度值
        归一化的原因是，假如是多目标优化，多个目标最好量纲/数量级统一
        由于时间越短越好，和适应度的定义相反，因此采取max-value的公式
        """
        max_fitness, min_fitness = self.solutions[-1].get_fitness, self.solutions[0].get_fitness
        for s in self.solutions:
            s.set_fitness((max_fitness - s.get_fitness())/(max_fitness-min_fitness))

    def add_solution(self, s: Solution):
        """有序列表，插入数据采用二分法定位最快，但是现在还没改成二分法，先记一下"""

        idx = 0
        for so in self.solutions:
            if s.get_fitness() > so.get_fitness():
                idx += 1
            else:
                break
        self.solutions.insert(idx, s)

    def get_rand_solution(self) -> Solution:
        return self.solutions[random.randint(0, len(self.solutions))]

