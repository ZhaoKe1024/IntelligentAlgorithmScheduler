#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/6 20:22
# @Author: ZhaoKe
# @File : ActivityGraph.py
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
from collections import deque
from datastructures.graph_entities import Vertex, Edge


class AOE(object):
    def __init__(self, vertex_list, edges):
        self.vertex_list = vertex_list

        self.__que = deque()
        self.__in_degree_list = [0 for _ in range(len(vertex_list))]
        # self.__out_degree_list = [0 for _ in range(len(vertex_list))]
        for ed in edges:
            self.__in_degree_list[ed.post_v] += 1

        self.adj_dag = [[] for _ in vertex_list]
        for edge in edges:
            self.adj_dag[edge.pre_v].append(vertex_list[edge.post_v])

    # 返回一个拓扑序列
    def topological_sort(self):
        V = len(self.vertex_list)
        print("count of vertex:", V)
        for i in range(V):
            if self.__in_degree_list[i] == 0:
                self.__que.append(self.vertex_list[i])
        res = []
        count = 0
        while len(self.__que) > 0:
            v = self.__que.popleft()
            print(v)
            res.append(v)
            count += 1
            self.__traverse_apply(v.index)
        if count < V:
            print("no topological order")
            return None
        else:
            return res

    def __traverse_apply(self, ind):
        for ver in self.adj_dag[ind]:
            self.__in_degree_list[ver.index] -= 1
            if self.__in_degree_list[ver.index] == 0:
                self.__que.append(ver)

    # 判断序列是不是该图的拓扑序列
    def check_path(self, path):
        assert len(path) == len(self.vertex_list), "length not matches."
        path_r = path[::-1]
        for i in range(len(path)):
            tmp = [ver.index for ver in self.adj_dag[path_r[i]]]
            for j in range(len(path) - i - 1):
                if self.vertex_list[path[j]].index in tmp:
                    return False
        return True


class AOV(object):
    def __init__(self, vertex_list, edges):
        self.vertex_list = vertex_list
        self.__in_degree_list = [0 for _ in range(len(vertex_list))]

        for ed in edges:
            self.__in_degree_list[ed.post_v] += 1

        self.adj_dag = [[] for _ in vertex_list]
        for edge in edges:
            self.adj_dag[edge.pre_v].append(vertex_list[edge.post_v])
        # for adj_link in self.adj_dag:
        #     print([ver.index for ver in adj_link])

        self.marked = [False for _ in range(len(vertex_list))]
        self.topological_set = []

    def key_path(self):
        """关键路径"""
        pass

    # 返回一个拓扑序列
    def topological_sort_all(self):
        self.topological_sort(deque())
        return self.topological_set

    def topological_sort(self, topo_vec):
        if len(topo_vec) == len(self.vertex_list):
            # print(topo_vec)
            # topo_vec.append(vertex)
            # print("here", self.vertex_list[i])
            # print(self.__in_degree_list)
            # print("result:", [ver_1.index for ver_1 in topo_vec])
            self.topological_set.append(list(topo_vec.copy()))
            # self.topological_set.append([ver_1.index for ver_1 in topo_vec])
            # return
        for i in range(len(self.vertex_list)):
            if self.__in_degree_list[i] == 0 and not self.marked[i]:
                self.marked[i] = True
                topo_vec.append(self.vertex_list[i])
                # print("push", [ver_1.index for ver_1 in topo_vec])
                self.__traverse_apply(self.vertex_list[i].index, False)

                self.topological_sort(topo_vec)

                self.__traverse_apply(self.vertex_list[i].index, True)
                topo_vec.pop()
                # print("pop", [ver_1.index for ver_1 in topo_vec])
                self.marked[i] = False

    def __traverse_apply(self, ind, add_or_sub=False):
        for ver in self.adj_dag[ind]:
            if add_or_sub:
                self.__in_degree_list[ver.index] += 1
            else:
                self.__in_degree_list[ver.index] -= 1

    # 判断序列是不是该图的拓扑序列
    def check_path(self, path):
        assert len(path) == len(self.vertex_list), "length not matches."
        path_r = path[::-1]
        for i in range(len(path)):
            tmp = [ver.index for ver in self.adj_dag[path_r[i]]]
            for j in range(len(path) - i - 1):
                if self.vertex_list[path[j]].index in tmp:
                    return False
        return True
