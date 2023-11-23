#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/23 20:41
# @Author: ZhaoKe
# @File : env_fjsp.py
# @Software: PyCharm
import random
import numpy as np

from fjspkits.fjsp_utils import read_Data_from_file

SEED = 1004
random.seed(SEED)
np.random.seed(SEED)


def create_fjsp():
    # list of jobs
    jobs, machine_num, task_num = read_Data_from_file("./datasets/fjsp_sets/brandimarte_mk03.txt")
    weights = np.array([1, 2, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 18, 20, 21])  # np.random.randint(1, 30, item_num)
    prices = np.array([1, 2, 3, 5, 7, 6, 10, 13, 15, 17, 18, 20, 22, 28, 35])  # np.random.randint(1, 99, item_num)
    capacity = 68  # np.random.randint(1, 99)
    item_num = weights.size
    init_state = np.zeros(item_num)
    init_action = 0  # NULL
    init_reward = 0
    return init_state, init_action, init_reward, weights, prices, capacity, item_num


class KnapsackEnv(object):

    def __init__(self):
        init_state, init_action, init_reward, weights, prices, capacity, item_num = create_knapsack()
        self.state = init_state
        self.action = init_action
        self.reward = init_reward
        self.weights = weights
        self.prices = prices
        self.capacity = capacity
        self.done = 0  # initialize 0 meaning that episode isn't finish
        self.item_num = item_num
        print('-------------------------------\ninit_state: ',
              init_state, '\nweights: ', weights, '\nprices: ', prices, '\ncapacity:', capacity, '\nitem_num: ',
              item_num,
              '\n-------------------------------')

    # get item_num
    def get_item_num(self):
        return self.item_num

    # get state after integer action
    def get_state(self, action):
        self.state[action] = 1
        return self.state

    # get state_space
    def get_state_space(self):
        return self.item_num

    # get action_space
    def get_action_space(self):
        return self.item_num

    # get reward about present state
    def get_reward(self):
        weight_sum = 0
        price_sum = 0
        for i in np.where(self.state == 1):
            for j in self.weights[i]:
                weight_sum += j
            for j in self.prices[i]:
                price_sum += j

        if weight_sum > self.capacity:
            self.done = 1  # episode finish
            return self.reward
        else:
            self.reward = price_sum
            return self.reward

    # get 1 step at env
    def step(self, action):
        self.state = self.get_state(action)
        self.reward = self.get_reward()
        return self.state, self.reward, self.done

    # reset env
    def reset(self):
        self.state = np.zeros(self.item_num)
        self.action = -100
        self.reward = 0
        self.done = 0
        return self.state


if __name__ == '__main__':
    create_fjsp()
