#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/4 10:51
# @Author: ZhaoKe
# @File : sortedlist.py
# @Software: PyCharm
"""
排序单链表，结点按值排序

SortedSinglyList<Node<Triple>>
SortedSinglyList.head -> Node<Triple>
Node.data -> Triple
Triple(pre, post, values)

list.data.value
SortedSinglyList.Node.Triple.value
"""
import functools

from datastructures.graph_entities import Edge


class SortedSinglyList(object):
    def __init__(self, values=None):
        self.head = Node(None, None)

        __rear = self.head
        if values:
            values = sorted(values, key=functools.cmp_to_key(cmp_triples))
            for value in values:
                __rear.next_node = Node(value, None)
                __rear = __rear.next_node

    def get(self, index):
        p = self.head.next_node
        for _ in range(index):
            if p is not None:
                p = p.next_node
        return p.value if (index > 0 and p is not None) else None

    def search(self, key):
        p = self.head.next_node
        while p:
            # print(p.data)
            if cmp_triples(key, p.data) == 0:
                return p
            elif cmp_triples(key, p.data) > 0:
                p = p.next_node
            else:
                return None
        # return None

    def insert(self, x):
        front = self.head
        p = front.next_node
        # if p is None:
        #     self.head.next_node = Node(x, None)
        #     return self.head.next_node
        while p and cmp_triples(x, p.data) > 0:
            front = p
            p = p.next_node
        front.next_node = Node(x, p)
        return front.next_node

    def set(self, i, x):
        p = self.head.next_node
        for j in range(i):
            if p is not None:
                p = p.next_node
        if i > 0 and p is not None:
            p.value = x

    def is_empty(self):
        return self.head.next_node is None

    def size(self):
        res = 0
        p = self.head.next_node
        while p:
            p = p.next_node
            res += 1
        return res

    def __str__(self):
        p = self.head.next_node
        if p is None:
            return "[]"
        res = '['+str(p.data)
        while p:
            p = p.next_node
            if p is not None:
                res += ', ' + str(p.data)
        return res+']'


class Node(object):
    def __init__(self, data, next_node):
        self.data = data
        self.next_node = next_node

    def __str__(self):
        return str(self.data.weight)


def cmp_triples(t1, t2):
    if t1.pre_v == t2.pre_v and t1.post_v == t2.post_v:
        return 0
    return -1 if t1.pre_v < t2.pre_v or (t1.pre_v == t2.pre_v and t1.post_v < t2.post_v) else 1


if __name__ == '__main__':
    sl = SortedSinglyList([Edge(1, 3, 4),
                           Edge(2, 6, 8),
                           Edge(2, 2, 3),
                           Edge(1, 8, 6),
                           Edge(3, 5, 4)])
    print(sl)
    print(sl.size())
    sl.insert(Edge(3,2,1))
    # print(cmp_triples(Triple(1, 8, 6), Triple(1, 8, 6)))
    # print(cmp_triples(Triple(1, 2, 6), Triple(1, 3, 4)))
    # print(cmp_triples(Triple(1, 8, 6), Triple(1, 2, 6)))
    # print(cmp_triples(Triple(3, 6, 8), Triple(2, 6, 8)))
    print(sl)
