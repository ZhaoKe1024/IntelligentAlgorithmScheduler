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
from collections import deque

from entities.AdjList_Graph import AdjListGraph, Vertex
from entities.graph_entities import Edge


class AOE(object):
    def __init__(self, vertex_list, edges):
        self.vertex_list = vertex_list
        self.DAG = AdjListGraph(vertex_list, edges)

        self.__que = deque()
        self.__in_degree_list = [0 for _ in range(len(vertex_list))]
        # self.__out_degree_list = [0 for _ in range(len(vertex_list))]
        for ed in edges:
            self.__in_degree_list[ed.post_v] += 1

    # 返回一个拓扑序列
    def topological_sort(self):
        V = self.DAG.vertex_count()
        print("count of vertex:", V)
        for i in range(V):
            if self.__in_degree_list[i] == 0:
                self.__que.append(self.DAG.get_vertex(i))
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

    def __traverse_apply(self, index):
        p = self.DAG.AdjMatrix.row_list[index].head.next_node
        while p:
            # print(p.data)
            self.__in_degree_list[p.data.post_v] -= 1
            if self.__in_degree_list[p.data.post_v] == 0:
                self.__que.append(self.vertex_list[p.data.post_v])
            p = p.next_node
        # return new_list

    # 判断序列是不是该图的拓扑序列
    def check_path(self, path):
        assert len(path) == len(self.vertex_list), "length not matches."
        path_r = path[::-1]
        # print("reverse path:", path_r)
        for i in range(len(path)):
            link = self.DAG.AdjMatrix.row_list[path_r[i]].head.next_node
            tmp = []
            while link:
                tmp.append(link.data.post_v)
                link = link.next_node
            # print(f"vertex {path_r[i]}, post: ", tmp)
            for j in range(len(path)-i-1):
                # print(self.vertex_list[path[j]].index)
                if self.vertex_list[path[j]].index in tmp:
                    return False
        return True


if __name__ == '__main__':
    # source 0, sink 8
    vertices1 = [Vertex(i, chr(65 + i)) for i in range(9)]
    edges1 = [
        Edge(0, 1, 6),
        Edge(0, 2, 4),
        Edge(0, 3, 5),
        Edge(1, 4, 1),
        Edge(2, 4, 1),
        Edge(3, 5, 2),
        Edge(4, 6, 9),
        Edge(4, 7, 7),
        Edge(5, 7, 4),
        Edge(6, 8, 2),
        Edge(7, 8, 4)
    ]
    graph = AOE(vertices1, edges1)
    res = graph.topological_sort()
    # for item in res:
    #     print(item.index, end=', ')
    # print()
    path1 = [0, 3, 1, 2, 5, 4, 7, 6, 8]
    print(graph.check_path(path1))
    path2 = [1, 2, 3, 0, 5, 6, 4, 7, 8]
    print(graph.check_path(path2))

