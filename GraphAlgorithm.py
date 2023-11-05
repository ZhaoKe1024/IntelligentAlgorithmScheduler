#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/5 17:52
# @Author: ZhaoKe
# @File : GraphAlgorithm.py
# @Software: PyCharm
""" AOE network
# 一种有向无环图，用有向边表示活动，顶点表示事件
# 工程至少需要多长时间？
# 哪些活动需要加快，方能缩短工期？
# 概念参考 https://blog.csdn.net/fangfanglovezhou/article/details/125230610

# # # AOV network
# 边上的活动 图结构: 常用于 APS 智能排程系统
# 求解关键路径(工程最短时间)
# 求解最早、最迟发生时间
"""
from entities.AdjList_Graph import AdjListGraph, Vertex
from entities.sortedlist import Triple


class AOE(object):
    def __init__(self, vertex_list, edges):
        self.DAG = AdjListGraph(vertex_list, edges)


if __name__ == '__main__':
    # source 0, sink 8
    vertices1 = [Vertex(i, chr(65 + i)) for i in range(8)]
    edges1 = [
        Triple(0, 1, 6),
        Triple(0, 2, 4),
        Triple(0, 3, 5),
        Triple(1, 4, 1),
        Triple(2, 4, 1),
        Triple(3, 5, 2),
        Triple(4, 6, 9),
        Triple(4, 7, 7),
        Triple(5, 7, 4),
        Triple(6, 8, 2),
        Triple(7, 8, 4)
    ]
    graph = AOE(vertices1, edges1)

