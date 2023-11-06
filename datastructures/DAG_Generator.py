#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/6 21:25
# @Author: ZhaoKe
# @File : DAG_Generator.py
# @Software: PyCharm
import numpy as np

from datastructures.graph_entities import Edge

"""
给定50个顶点
DAG最多可以有多少个边？N(N-1)/2
"""
def genetate_dag():
    VERTEX_NUM = 50
    MAX_VERTEX_NUM = VERTEX_NUM * (VERTEX_NUM-1) // 2
    bl = np.floor(0.75 * VERTEX_NUM + 0.25 * MAX_VERTEX_NUM)
    up = np.floor(0.25 * VERTEX_NUM + 0.75 * MAX_VERTEX_NUM)

    edge_num = np.random.randint(bl, up)

    vertices = list(range(VERTEX_NUM))
    edges = []
    count = 0
    for i in range(VERTEX_NUM):
        start = vertices[i]

        step_num = np.random.randint(4, 7)
        if count + step_num > edge_num:
            step_num = edge_num - count
        count += step_num
        for _ in range(step_num):
            end = np.random.randint(start + 4, start + 7)
            if end > MAX_VERTEX_NUM:
                end = MAX_VERTEX_NUM-1
                break
            edges.append(Edge(start, end, np.random.randint(20, 50)))
    with open(f"../datasets/graph_example/dag2-v{VERTEX_NUM}-e{len(edges)}.txt", 'w') as fo:
        for edge in edges:
            print(count, edge)
            fo.write(f"{edge.pre_v},{edge.post_v},{edge.weight}\n")


if __name__ == '__main__':
    genetate_dag()
