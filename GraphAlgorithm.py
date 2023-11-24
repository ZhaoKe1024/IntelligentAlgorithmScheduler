#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/5 17:52
# @Author: ZhaoKe
# @File : GraphAlgorithm.py
# @Software: PyCharm
from datastructures.ActivityGraph import ActivityNetwork
from datastructures.graph_entities import Vertex, Edge


def AOV_example():
    # source 0, sink 8
    indices = set()
    edges = []
    with open("datasets/graph_example/dag1.txt") as fi:
        line = fi.readline()
        idx = 0
        while line:
            parts = line.strip().split(',')
            indices.add(int(parts[0]))
            indices.add(int(parts[1]))
            edges.append(Edge(edge_id=idx, pre=int(parts[0]), post=int(parts[1]), weight=int(parts[2])))
            line = fi.readline()
            idx += 1
    vertices1 = [Vertex(i, chr(65 + i)) for i in indices]
    graph = ActivityNetwork(vertices1, edges)
    single_topo_orders = graph.topological_sort_rand()
    print([ver.index for ver in single_topo_orders])


def AOE_example():
    # source 0, sink 8
    indices = set()
    edges = []
    with open("datasets/graph_example/dag3.txt") as fi:
        line = fi.readline()
        while line:
            parts = line.strip().split(',')
            indices.add(int(parts[0]))
            indices.add(int(parts[1]))
            edges.append(Edge(int(parts[0]), int(parts[1]), int(parts[2])))
            line = fi.readline()
    vertices1 = [Vertex(i, chr(65 + i)) for i in indices]
    aoe_dag = ActivityNetwork(vertices1, edges)

    for edge in edges:
        print(f"{edge.pre_v},{edge.post_v},{edge.weight}")
    # topo_list = aoe_dag.topological_sort()
    # print([ver.index for ver in topo_list])
    print('----critical path----')
    cp = aoe_dag.critical_path()
    print(",".join([str(item) for item in cp]))


if __name__ == '__main__':
    print("-------AOV---------")
    AOV_example()
