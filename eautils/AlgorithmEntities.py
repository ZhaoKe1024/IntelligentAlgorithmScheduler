# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2021-04-22 18:32
import numpy as np


# binary particle
# 粒子代表问题的一个解，一个解决方案，由01标识，表示粒子的位置向量，另外还有速度向量
# 使用取整求余的方法解决粒子群的离散应用问题，因此这里速度还是原来的含义，而位置的含义变为任务在哪个计算节点上执行
class DParticle:
    def __init__(self, length: int, fitness: float = 0):
        self.solution = np.array([0 for _ in range(length)], dtype=int)  # 获得一个长度为size的01随机向量
        self.velocity = np.array([0 for _ in range(length)], dtype=float)  # 获得一个长度为size的零向量
        self.fitness = fitness

    def __str__(self):
        result = [str(e) for e in self.solution]  # 例如['0', '1', '0', '1', '1', '1', '0', '0']
        return '[' + ', '.join(result) + ']'  # 例如'[0,1,0,1,1,1,0,0]'
