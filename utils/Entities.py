# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2021-03-06 16:12
import numpy as np


class Cloudlet:
    def __init__(self, cpu_demand: float, mem_demand: float, hd_demand, bw_demand):
        self.cpu_demand = cpu_demand
        self.mem_demand = mem_demand
        # # 100M ~ 2000M
        self.hd_demand = hd_demand
        self.bw_demand = bw_demand

    def getData(self):
        return [self.cpu_demand, self.mem_demand]  # + "," + str(self.hd_demand)

    def __str__(self):
        return ("Cloudlet: cpu:" + str(self.cpu_demand) + ", mem:" + str(self.mem_demand)
                + ", hd:" + str(self.hd_demand)
                + ", bw:" + str(self.bw_demand))

    def str_as_row(self):
        return f"{self.cpu_demand},{self.mem_demand},{self.hd_demand},{self.bw_demand}"


class VM:
    def __init__(self, vm_id: int, cpu_supply: float, cpu_velocity: float, mem_supply: float, mem_capacity: float,
                 hd_supply=0., hd_capacity=0., bw_supply=0., bw_capacity=0.):
        self.id = vm_id
        self.cpu_supply = cpu_supply
        self.cpu_velocity = cpu_velocity
        self.mem_supply = mem_supply
        self.mem_capacity = mem_capacity
        self.hd_supply = hd_supply
        self.hd_capacity = hd_capacity
        self.bw_supply = bw_supply
        self.bw_capacity = bw_capacity

    def getData(self):
        return [self.id, self.cpu_supply, self.cpu_velocity, self.mem_supply, self.mem_capacity  # \
            , self.hd_supply, self.hd_capacity
            , self.bw_supply, self.bw_capacity]

    def __str__(self):
        return "Node:" + str(self.id) + "--cpu:" + str(self.cpu_supply) + "/" + str(self.cpu_velocity) \
            + ", mem:" + str(self.mem_supply) + "/" + str(self.mem_capacity) \
            + ", hd:" + str(self.hd_supply) + "/" + str(self.hd_capacity) \
            + ", bw:" + str(self.bw_supply) + "/" + str(self.bw_capacity)

    def str_as_row(self):
        return (f"{self.id},{self.cpu_supply},{self.cpu_velocity},{self.mem_supply},{self.mem_capacity},"
                + f"{self.hd_supply},{self.hd_capacity},"
                + f"{self.bw_supply},{self.bw_capacity}")


# 评价函数
def evaluate_particle(p, cloudlets, vms) -> int:
    cpu_util = np.zeros(len(vms))
    mem_util = np.zeros(len(vms))
    # hd_util = np.zeros(len(vms))

    for i in range(len(vms)):
        cpu_util[i] = vms[i].cpu_supply
        mem_util[i] = vms[i].mem_supply
        # hd_util[i] = vms[i].hd_supply

    for i in range(len(cloudlets)):
        cpu_util[p[i]] += cloudlets[i].cpu_demand
        mem_util[p[i]] += cloudlets[i].mem_demand
        # hd_util[p[i]] += cloudlets[i].hd_demand

    for i in range(len(vms)):
        # print(cpu_util)
        # print(mem_util)
        # print(hd_util)
        if cpu_util[i] > vms[i].cpu_velocity:
            return 100
        if mem_util[i] > vms[i].mem_capacity:
            return 100
        # if hd_util[i] > vms[i].hd_capacity:
        #     return 100

    for i in range(len(vms)):
        cpu_util[i] /= vms[i].cpu_velocity
        mem_util[i] /= vms[i].mem_capacity
        # hd_util[i] /= vms[i].hd_capacity

    # # print(np.std(hd_util, ddof=1))
    # if np.std(hd_util, ddof=1) is None:
    #     print("ok")
    #     return np.std(cpu_util, ddof=1) + np.std(mem_util, ddof=1)
    # print(np.std(hd_util, ddof=1))

    # 个体评估函数必须选择下面的其中一个，因为在资源个数较少的时候，0的标准差为nan，导致迭代不会有效果
    # The individual evaluation function must choose one of the following,
    # because when the number of resources is small, the standard deviation of 0 is Nan,
    # so the iteration will not have effect
    return np.std(cpu_util, ddof=1) + np.std(mem_util, ddof=1)
    # return np.std(cpu_util, ddof=1) * np.std(mem_util, ddof=1) * np.std(hd_util, ddof=1)


def calculate_fitness(p, cloudlets, vms) -> float:
    # print(evaluate_particle(p, cloudlets, vms), ' ', 1 / evaluate_particle(p, cloudlets, vms))
    # return evaluate_particle(p, cloudlets, vms)

    # 适应度函数必须按照如下的方式定义，因为所有的进化算法的选择阶段都是按照选择适应度高的个体定义的，
    # 适应度函数如果不这样定义，迭代就不会有效果
    # The fitness function must be defined in the following way,
    # because the selection stage of all evolutionary algorithms is defined
    # according to the selection of individuals with high fitness.
    # If the fitness function is not defined in this way, the iteration will not be effective
    return 1 / evaluate_particle(p, cloudlets, vms)


