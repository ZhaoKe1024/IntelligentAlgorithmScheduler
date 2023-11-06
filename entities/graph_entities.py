#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/6 11:40
# @Author: ZhaoKe
# @File : graph_entities.py
# @Software: PyCharm
import numpy as np


class Vertex(object):
    def __init__(self, index, name):
        self.index = index
        self.name = name
        # vertex as activity
        self.dur = None

    def __str__(self):
        return f"vertex:{self.index}, v_name:{self.name}"


class Edge(object):
    def __init__(self, pre, post, weight=np.inf):
        self.pre_v = pre
        self.post_v = post
        self.weight = weight

    def __str__(self):
        return f"{self.pre_v}->{self.post_v}:{self.weight}"