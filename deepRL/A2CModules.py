#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/22 8:32
# @Author: ZhaoKe
# @File : A2CModules.py
# @Software: PyCharm
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
TD_ERROR_EPSILON = 0.0001


class RolloutStorage(object):  # 用于存储用于Advantage学习的Rollout信息
    def __init__(self, n_states, gamma):
        self.memory = deque(maxlen=n_states)
        self.dis_rewards = deque(maxlen=n_states)
        self.GAMMA = gamma

    def push(self, *args):
        if self.memory.maxlen == len(self.memory):
            print("The error occurs : it is already full")
        self.memory.append(Transition(*args))

    def compute_returns(self, next_value):  # calculating discounted rewards
        dis_reward = next_value
        for ad_step in reversed(range(len(self.memory))):  # [注意]从最后开始按逆序计算（小插曲结束后学习的原因）
            dis_reward = dis_reward * self.GAMMA + self.memory[ad_step][2]
            self.dis_rewards.appendleft(dis_reward)

    def clear(self):
        self.memory.clear()
        self.dis_rewards.clear()

    def __len__(self):
        return len(self.memory)


