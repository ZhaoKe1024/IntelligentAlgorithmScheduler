#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/16 16:24
# @Author: ZhaoKe
# @File : APSonGraph.py
# @Software: PyCharm

from datastructures.ActivityGraph import AOV, AOE
from datastructures.graph_entities import Vertex, Edge

from fjspkits.fjsp_utils import read_Data_from_json


class SOP(object):
    def __init__(self, idx, task_id, timespan):
        self.index = idx
        self.taskId = task_id
        self.preSOP = []
        self.next = []
        self.timespan = timespan



def run():
    read_Data_from_json("./datasets/fjsp_sets/排程测试入参01.json")
    # # source 0, sink 8
    # indices = set()
    # edges = []
    # with open("datasets/graph_example/dag1.txt") as fi:
    #     line = fi.readline()
    #     while line:
    #         parts = line.strip().split(',')
    #         indices.add(int(parts[0]))
    #         indices.add(int(parts[1]))
    #         edges.append(Edge(int(parts[0]), int(parts[1]), int(parts[2])))
    #         line = fi.readline()
    # vertices1 = [Vertex(i, chr(65 + i)) for i in indices]
    #
    # for edge in edges:
    #     print(f"{edge.pre_v},{edge.post_v},{edge.weight}")
    # graph = AOV(vertices1, edges)
    # all_topo_orders = graph.topological_sort_all()
    #
    # path1 = [0, 3, 1, 2, 5, 4, 7, 6, 8]
    # print("is it path topological ordered:")
    # print(graph.check_path(path1))
    # # path2 = [1, 2, 3, 0, 5, 6, 4, 7, 8]
    # # print(graph.check_path(path2))
    # print("all topological order of this DAG:")
    # for topo in all_topo_orders:
    #     print([ver.index for ver in topo])


if __name__ == '__main__':
    run()
