#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/4 10:34
# @Author: ZhaoKe
# @File : LinkedMatrix.py
# @Software: PyCharm
"""
链表型邻接矩阵，特点是稀疏
每一行都是一个SortedSinglyList
"""
from datastructures.sortedlist import SortedSinglyList
from datastructures.graph_entities import Edge


class LinkedMatrix(object):
    def __init__(self, m, n, elems):
        self.rows = m
        self.columns = n
        self.row_list = [SortedSinglyList() for _ in range(m)]
        if elems:
            for elem in elems:
                self.set_triple(elem)

    def set(self, i, j, x):
        if 0 <= i < self.rows and 0 <= j < self.columns:
            link = self.row_list[i]
            tri = Edge(i, j, x)
            find = link.search(tri)
            if find:
                find.value = x
            else:
                link.insert(tri)

    def set_triple(self, tri):
        self.set(tri.pre_v, tri.post_v, tri.weight)

    def print_matrix(self):
        for row_list_item in self.row_list:
            print(row_list_item)


if __name__ == '__main__':
    elems = [
        Edge(0, 1, 45),
        Edge(0, 2, 28),
        Edge(0, 3, 10),
        Edge(1, 0, 45),
        Edge(2, 0, 28),
        Edge(1, 2, 12),
        Edge(1, 4, 21),
        Edge(3, 0, 10),
        Edge(2, 1, 12),
        Edge(2, 4, 26),
        Edge(3, 2, 17),
        Edge(3, 4, 15),
        Edge(4, 1, 21),
        Edge(4, 3, 15),
        Edge(4, 2, 26),
    ]
    lm = LinkedMatrix(5, 5, elems)
    lm.print_matrix()
