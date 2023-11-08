#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/8 12:01
# @Author: ZhaoKe
# @File : jsp_model.py
# @Software: PyCharm
"""
JSSP Job Shop Scheduling Problem
FJSP Flexible JobShop Scheduling Problem

"""


class JSSP_Optimizer(object):
    def __init__(self, data):
        self.data = data
        self.solution_factory = SolutionGenerator(data=data)


class JSPSolution(object):
    def __init__(self):
        pass

    def init_solution(self):
        pass


class SolutionGenerator(object):
    def __init__(self, data):
        self.jssp_instance = data
