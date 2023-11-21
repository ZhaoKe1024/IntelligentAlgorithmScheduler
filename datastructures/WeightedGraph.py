#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/20 8:55
# @Author: ZhaoKe
# @File : WeightedGraph.py
# @Software: PyCharm
import numpy as np


class StateVertex(object):
    def __init__(self, idx, next_list=None):
        self.idx = idx
        self.next_list = next_list

    def __str__(self):
        return f"State:{self.idx} {[str(item) for item in self.next_list]}"


class EdgeReward(object):
    def __init__(self, pre, post, weight):
        self.pre_s = pre
        self.post_s = post
        self.weight = weight

    def __str__(self):
        return f"[{self.pre_s}-{self.weight}->{self.post_s}]"


Q = np.zeros((6, 6))
R_list = [StateVertex(0, [EdgeReward(0, 4, 0)]),
          StateVertex(1, [EdgeReward(1, 3, 0), EdgeReward(1, 5, 100)]),
          StateVertex(2, [EdgeReward(2, 3, 0)]),
          StateVertex(3, [EdgeReward(3, 1, 0), EdgeReward(3, 2, 0), EdgeReward(3, 4, 0)]),
          StateVertex(4, [EdgeReward(4, 0, 0), EdgeReward(4, 3, 0), EdgeReward(4, 5, 100)]),
          StateVertex(5, [EdgeReward(5, 1, 0), EdgeReward(5, 4, 0), EdgeReward(5, 5, 100)])]

gamma = 0.8
# for rew in R_list:
#     print(rew)

for step in range(1000):
    c_state = np.random.randint(0, len(R_list))
    # print(f"随机选择一个状态:{c_state}")
    act_list = R_list[c_state].next_list
    # print("该状态的下一时刻选择：", [str(item) for item in act_list])
    c = np.random.randint(0, len(act_list))
    # print(f"随机选择下一时刻状态:{c}")
    action = act_list[c]
    # print("该状态的描述", str(action))
    # print("该状态的对应Q行", Q[action.post_s,])
    # print("对应Q行的最大值：", np.max(Q[action.post_s,]))
    # print(np.where(Q[action.post_s,] == np.max(Q[action.post_s,])))
    max_index = np.where(Q[action.post_s,] == np.max(Q[action.post_s,]))[0]
    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size=1))
    else:
        max_index = int(max_index)
    max_value = Q[action.post_s, max_index]
    Q[c_state, action.post_s] = R_list[c_state].next_list[c].weight + gamma * max_value

res = Q / np.max(Q) * 100
for row in res:
    print([np.round(item, 2) for item in row])

STATE_START = 2
STATE_END = 5
step = [STATE_START]
while STATE_START != STATE_END:
    max_index = np.where(Q[STATE_START,] == np.max(Q[STATE_START,]))[0]
    if max_index.shape[0] > 1:
        next_step_index = int(np.random.choice(max_index, size=1))
    else:
        next_step_index = int(max_index)
    step.append(next_step_index)
    STATE_START = next_step_index

print(step)
