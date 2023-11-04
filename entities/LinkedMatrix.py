#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/4 10:34
# @Author: ZhaoKe
# @File : LinkedMatrix.py
# @Software: PyCharm
from entities.sortedlist import SortedSinglyList, Triple


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
            tri = Triple(i, j, x)
            find = link.search(tri)
            if find:
                find.value = x
            else:
                link.insert(tri)

    def set_triple(self, tri):
        self.set(tri.row, tri.column, tri.value)

    def print_matrix(self):
        for row_list_item in self.row_list:
            print(row_list_item)


if __name__ == '__main__':
    elems = [
        Triple(0,1,45),
        Triple(0,2,28),
        Triple(0,3,10),
        Triple(1,0,45),
        Triple(2,0,28),
        Triple(1,2,12),
        Triple(1,4,21),
        Triple(3,0,10),
        Triple(2,1,12),
        Triple(2,4,26),
        Triple(3,2,17),
        Triple(3,4,15),
        Triple(4,1,21),
        Triple(4,3,15),
        Triple(4,2,26),
    ]
    lm = LinkedMatrix(5, 5, elems)
    lm.print_matrix()
