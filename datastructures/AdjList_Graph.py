#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/4 9:06
# @Author: ZhaoKe
# @File : AdjList_Graph.py
# @Software: PyCharm
"""
Graph Structure based on Adjacent Matrix
functions:
1 depth first traverse
2 breadth first traverse
3 Dijkstra, Floyd,
application:
1. AOE activity on edges
2. AOV activity on vertices
"""
import random
from collections import deque

import numpy as np

from datastructures.LinkedMatrix import LinkedMatrix
from datastructures.graph_entities import Edge, Vertex


class AdjListGraph(object):
    def __init__(self, vertex_list, edges):
        self._vertex_list = vertex_list
        self._edge_list = edges
        self.AdjMatrix = LinkedMatrix(len(vertex_list), len(vertex_list), edges)
        self.MAX_WEIGHT = np.inf


    def DFS_Traverse(self, index):
        visited = [False for _ in range(self.vertex_count())]
        j = index
        while True:
            if not visited[j]:
                print('{', end='')
                self.__depth_fs(j, visited)
                print('}')
            j = (j + 1) % self.vertex_count()
            if j == index:
                break
        print()

    def __depth_fs(self, i, visited):
        print(self.get_vertex(i), end=' ')
        visited[i] = True
        j = self.next(i, -1)
        while j != -1:
            # print("DFS", j)
            if not visited[j]:
                self.__depth_fs(j, visited)
            j = self.next(i, j)

    def BFS_Traverse(self, i):
        visited = [False for _ in range(self.vertex_count())]
        j = i
        while True:
            if not visited[j]:
                print('{', end='')
                self.__breadth_fs(j, visited)
                print('}')
            j = (j + 1) % self.vertex_count()
            if j == i:
                break
        print()

    def __breadth_fs(self, i, visited):
        print(self.get_vertex(i), end=' ')
        visited[i] = True
        que = deque()
        que.append(i)
        while len(que) > 0:
            # print(que)
            i = que.popleft()
            j = self.next(i, -1)
            while j != -1:
                print(self.get_vertex(j), end=' ')
                visited[j] = True
                que.append(j)
                j = self.next(i, j)

    def next(self, i, j):
        n = self.vertex_count()
        if (
                0 < i < n and -1 < j != i):
            link = self.AdjMatrix.row_list[i]
            find = link.head.next_node
            if j == -1:
                return find.data.column if (find is not None) else -1
            find = link.search(Edge(i, j, 0))
            if find is not None:
                find = find.next_node
                if find is not None:
                    return find.data.column
        return -1

    def get_vertex(self, index):
        return self._vertex_list[index]

    def set_vertex(self, x):
        self._vertex_list.append(x)

    def vertex_count(self):
        return len(self._vertex_list)



if __name__ == '__main__':
    print("OK")
    vertices0 = [Vertex(i, chr(65 + i)) for i in range(10)]
    edges0 = [
        Edge(0, 1, 3),
        Edge(0, 2, 2),
        Edge(1, 3, 2),
        Edge(2, 3, 4),
        Edge(1, 4, 3),
        Edge(2, 5, 3),
        Edge(3, 5, 2),
        Edge(4, 5, 1),
    ]
    vertices1 = [Vertex(i, chr(65 + i)) for i in range(10)]
    edges1 = [
        Edge(0, 1, 5),
        Edge(0, 2, 6),
        Edge(1, 3, 3),
        Edge(2, 3, 6),
        Edge(2, 4, 3),
        Edge(3, 4, 3),
        Edge(3, 5, 4),
        Edge(3, 6, 5),
        Edge(4, 6, 1),
        Edge(4, 7, 10),
        Edge(5, 9, 4),
        Edge(6, 8, 5),
        Edge(7, 8, 2),
        Edge(8, 9, 2)
    ]
    random.shuffle(edges1)
    for edge in edges1:
        print(edge)
    graph = AdjListGraph(vertices1, edges1)
    print("----Graph----")
    for item in graph.AdjMatrix.row_list:
        print(item)
    print("----DFS----")
    graph.DFS_Traverse(6)
    print("----BFS----")
    graph.BFS_Traverse(3)
