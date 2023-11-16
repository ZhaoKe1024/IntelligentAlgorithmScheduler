#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/16 16:24
# @Author: ZhaoKe
# @File : APSonGraph.py
# @Software: PyCharm
from datastructures.ActivityGraph import ActivityNetwork
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
    json_data = read_Data_from_json("./datasets/fjsp_sets/排程测试入参01.json")
    vertex_list = []
    command_id2ver_id = dict()
    ver_id = 0
    for dict_item in json_data:
        for key in dict_item:
            # print(key, ':', dict_item[key])
            command_id2ver_id[dict_item["id"]] = ver_id
            vertex_list.append(Vertex(index=ver_id, name=dict_item["id"], duration=int(dict_item["time"])))
            ver_id += 1
    edge_list = []
    ver_id, e_id = 0, 0
    for dict_item in json_data:
        for _ in dict_item:
            if dict_item["nextCommandIds"] is not None:
                for post_v in dict_item["nextCommandIds"]:
                    edge_list.append(Edge(e_id, ver_id, int(command_id2ver_id[post_v])))
                    e_id += 1
            if dict_item["preCommandIds"] is not None:
                for pre_v in dict_item["preCommandIds"]:
                    edge_list.append(Edge(e_id, int(command_id2ver_id[pre_v]), ver_id))
                    e_id += 1
            ver_id += 1
    print("number of vertices: ", len(vertex_list))
    print("number of edges", len(edge_list))
    # for edge in edge_list:
    #     print(f"{edge.pre_v},{edge.post_v}")
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
    graph = ActivityNetwork(vertex_list, edge_list)
    graph.topological_sort_all()
    # all_topo_orders = graph.topological_sort_single()
    # for v in all_topo_orders:
    #     print(v.index)
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
