#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/4 9:06
# @Author: ZhaoKe
# @File : AdjList_Graph.py
# @Software: PyCharm
from collections import deque

import numpy as np

from entities.LinkedMatrix import Triple, LinkedMatrix


class AdjListGraph(object):
    def __init__(self, vertex_list, edges):
        self._vertex_list = vertex_list
        self._edge_list = edges
        self.adj_list = LinkedMatrix(len(vertex_list), len(vertex_list), edges)
        self._adj_list = None
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
            j = (j+1) % self.vertex_count()
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
            link = self._adj_list.row_list[i]
            find = link.head.next_node
            if j == -1:
                return find.data.column if (find is not None) else -1
            find = link.search(Triple(i, j, 0))
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


class Vertex(object):
    def __init__(self, index, name):
        self.index = index
        self.name = name

    def __str__(self):
        return f"vertex:{self.index}, v_name:{self.name}"


class Edge(object):
    def __init__(self, pre, post, weight=np.inf):
        self.pre_v = pre
        self.post_v = post
        self.weight = weight


if __name__ == '__main__':
    print("OK")
    vertices = [Vertex(i, chr(65 + i)) for i in range(6)]
    edges = [
        Triple(0, 1, 3),
        Triple(0, 2, 2),
        Triple(1, 3, 2),
        Triple(2, 3, 4),
        Triple(1, 4, 3),
        Triple(2, 5, 3),
        Triple(3, 5, 2),
        Triple(4, 5, 1),
    ]
    graph = AdjListGraph(vertices, edges)
    print("----Graph----")
    for item in graph.adj_list.row_list:
        print(item)
    print("----DFS----")
    graph.DFS_Traverse(0)
    # print("----BFS----")
    # graph.BFS_Traverse(0)
