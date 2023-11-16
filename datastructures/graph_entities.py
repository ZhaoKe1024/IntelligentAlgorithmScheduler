#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/6 11:40
# @Author: ZhaoKe
# @File : graph_entities.py
# @Software: PyCharm
import numpy as np


class Vertex(object):
    def __init__(self, index, name, duration=None):
        self.index = index
        self.name = name
        # vertex as activity
        self.duration = duration
        self.in_degree = 0
        self.out_degree = 0

    def __str__(self):
        return f"vertex:{self.index}(v_name:{self.name})"

    def __eq__(self, other):
        return self.index == other.index


class Edge(object):
    def __init__(self, edge_id, pre, post, weight=np.inf):
        self.edge_id = edge_id
        self.pre_v = pre
        self.post_v = post
        self.weight = weight

    def __str__(self):
        return f"{self.pre_v}->{self.post_v}:{self.weight}"

    def __eq__(self, other):
        return self.pre_v == other.pre_v and self.post_v == other.post_v
