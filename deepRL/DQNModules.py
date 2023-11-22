#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/20 18:28
# @Author: ZhaoKe
# @File : DQNModules.py
# @Software: PyCharm
import random
import numpy as np
from collections import deque, namedtuple


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
TD_ERROR_EPSILON = 0.0001


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        if self.memory.maxlen == len(self.memory):
            self.memory.pop()  # 리워드 낮은 값 삭제
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PrioritizedMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, td_error):
        if self.memory.maxlen == len(self.memory):
            td_error = self.memory.pop()  # 리워드 낮은 값 삭제
            del td_error
        self.memory.append(td_error)

    def get_prioritized_indexes(self, batch_size):  # TD 오차에 따른 확률로 인덱스를 추출
        sum_absolute_td_error = np.sum(np.absolute(self.memory))  # TD 오차의 합을 계산
        sum_absolute_td_error += TD_ERROR_EPSILON * len(self.memory)  # 충분히 작은 값을 더해줌
        # batch_size 개만큼 난수를 생성하고 오름차순으로 정렬
        rand_list = np.random.uniform(0, sum_absolute_td_error, batch_size)
        rand_list = np.sort(rand_list)
        # 위에서 만든 난수로 인덱스를 결정
        indexes = []
        idx = 0
        tmp_sum_absolute_td_error = 0
        for rand_num in rand_list:
            while tmp_sum_absolute_td_error < rand_num:
                tmp_sum_absolute_td_error += (abs(self.memory[idx]) + TD_ERROR_EPSILON)
                idx += 1
                # TD_ERROR_EPSILON을 더한 영향으로 인덱스가 실제 갯수를 초과했을 경우를 위한 보정
                if idx >= len(self.memory):
                    idx = len(self.memory) - 1
            indexes.append(idx)
        return indexes

    def sample(self, batch_size, replay_memory):  # 리워드가 높은 케이스 중에서, TD-error에 따라 샘플을 추출한다.
        sample_idxes = self.get_prioritized_indexes(int(batch_size))
        return deque([replay_memory.memory[n] for n in sample_idxes])

    def clear(self):
        if len(self.memory) == 0:
            td_error = self.memory.pop()  # 리워드 낮은 값 삭제
            del td_error

    def __len__(self):
        return len(self.memory)
