#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/15 20:53
# @Author: ZhaoKe
# @File : fjsp_struct.py
# @Software: PyCharm
import random

import numpy as np
import operator
from fjspkits.fjsp_utils import calculate_exetime_load


class Solution(object):
    def __init__(self, machines, job_num, check=True):
        self.__machines = machines
        self.__fitness = -1.0
        self.job_num = job_num
        self.src_value = None
        if (not check) and (machines is not None):
            self.src_value, aligned_machines = calculate_exetime_load(machines, job_num)
            self.__machines = aligned_machines
            # print("time:", self.__fitness)

    def vectorize(self):
        machine_indices = [0]
        job_indices = []


    def get_machines(self):
        return self.__machines

    def get_fitness(self, max_value=None, min_value=None):
        # print("len of machines:", len(self.__machines))
        if self.__machines is None:
            return -1.0
        # print(self.__fitness)
        if self.__fitness <= -0.1:
            self.src_value, aligned_machines = calculate_exetime_load(self.__machines, self.job_num)
            if max_value - min_value < 0.00001:
                self.__fitness = (max_value - self.src_value) / max_value
            else:
                self.__fitness = (max_value-self.src_value)/(max_value-min_value)
            self.__machines = aligned_machines
        return self.__fitness

    def set_fitness(self, value):
        self.__fitness = value


class SolutionSortedList(object):
    """为了优化进化计算中的自然选择算子，用排序列表来组织种群是最好的
    desc: 最初的顺序
    """

    def __init__(self):
        self.solutions = []
        self.max_value = None
        self.min_value = None
        # self.desc = desc

    def update_fitness(self):
        """ 更新之后一定是降序
        初始化适应度值
        归一化的原因是，假如是多目标优化，多个目标最好量纲/数量级统一
        由于时间越短越好，和适应度的定义相反，因此采取max-value的公式
        """
        min_value, max_value = self.solutions[0].src_value, self.solutions[-1].src_value
        self.min_value, self.max_value = min_value, max_value
        # print(min_fitness, max_fitness)

        if max_value == min_value:
            if min_value <= 0:
                for s in self.solutions:
                    s.set_fitness(0.0)
            else:
                for s in self.solutions:
                    s.set_fitness(min_value)

        else:
            for s in self.solutions:
                s.set_fitness((max_value-s.src_value) / (max_value - min_value))

    def get_max_min_value(self):
        return self.solutions[-1].src_value, self.solutions[0].src_value

    def add_solution(self, s: Solution, desc=True):
        """有序列表，插入数据采用二分法定位最快，但是现在还没改成二分法，先记一下"""

        idx = 0
        for so in self.solutions:
            if desc:
                if s.get_fitness(self.max_value, self.min_value) < so.get_fitness(self.max_value, self.min_value):
                    idx += 1
                else:
                    break
            else:
                if s.src_value > so.src_value:
                    idx += 1
                else:
                    break
        self.solutions.insert(idx, s)

    def get_rand_solution(self) -> Solution:
        # print(len(self.solutions), random.randint(0, len(self.solutions)))
        return self.solutions[random.randint(0, len(self.solutions)-1)]
