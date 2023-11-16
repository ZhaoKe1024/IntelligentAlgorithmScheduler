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
        while line:
            parts = line.strip().split(',')
            indices.add(int(parts[0]))
            indices.add(int(parts[1]))
            edges.append(Edge(int(parts[0]), int(parts[1]), int(parts[2])))
            line = fi.readline()
    vertices1 = [Vertex(i, chr(65 + i)) for i in indices]

    for edge in edges:
        print(f"{edge.pre_v},{edge.post_v},{edge.weight}")
    graph = ActivityNetwork(vertices1, edges)
    all_topo_orders = graph.topological_sort_all()

    path1 = [0, 3, 1, 2, 5, 4, 7, 6, 8]
    print("is it path topological ordered:")
    print(graph.check_path(path1))
    # path2 = [1, 2, 3, 0, 5, 6, 4, 7, 8]
    # print(graph.check_path(path2))
    print("all topological order of this DAG:")
    for topo in all_topo_orders:
        print([ver.index for ver in topo])
        # print(graph.check_path([ver.index for ver in topo]))


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
    print("----AOE--------")
    AOE_example()

    # print(list(range(9, 0, -1)))
