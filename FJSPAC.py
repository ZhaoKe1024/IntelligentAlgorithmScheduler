# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2021-03-16 11:41
"""
柔性作业车间调度问题
(Flexible Job-shop Scheduling Problem, FJSP)
"""
import numpy as np
import random
from typing import List
from matplotlib import pyplot as plt


class ACFJSSP:
    def __init__(self):
        # 作业数，统一工序数，机器数
        self.job_num = 4
        self.process_num = 3
        self.machine_num = 6
        # 迭代次数
        self.iteration_num = 40
        # 蚂蚁数量
        self.ant_num = 30

        # 4个Job的3个工序在6台机器上的加工时间
        self.times = [
            [
                [2, 3, 4, None, None, None],  # 工件1第一道工序的三种选择
                [None, 3, None, 2, 4, None],
                [1, 4, 5, None, None, None]
            ],
            [
                [3, None, 5, None, 2, None],
                [4, 3, None, 6, None, None],
                [None, None, 4, None, 7, 11]  # 工件2第三道工序的三种选择
            ],
            [
                [5, 6, None, None, None, None],
                [None, 4, None, 3, 5, None],
                [None, None, 13, None, 9, 12]
            ],
            [
                [9, None, 7, 9, None, None],
                [None, 6, None, 4, None, 5],  # 工件4第二道工序的三种选择：机床2用6小时或者机床4用4小时或机床6用5小时
                [1, None, 3, None, None, 3]
            ]
        ]

        # 拓扑序的信息素浓度，初始值100
        # 12行4列
        # 表示因为一共有12个工序，每个工序对应哪个工件？
        self.topo_phs = [
            [100 for _ in range(self.job_num)]
            for _ in range(self.job_num * self.process_num)
        ]
        print(self.topo_phs)
        # 每个Job的每个工序的信息素浓度，初始值100
        # 4页3行6列
        # 表示任务的工序-机器对应？
        self.machine_phs = [
            [
                [100 for _ in range(self.machine_num)]
                for _ in range(self.process_num)
            ]
            for _ in range(self.job_num)
        ]
        print(self.machine_phs)


    def gen_topo_jobs(self) -> List[int]:
        """
        生成拓扑序
        Job在时空上处理的的拓扑序(Job索引)，这个序不能体现工序选择的机器
        :return 如[0,1,0,2,2,...]表示p11,p21,p12,p31,p32,...
        """
        # 按照每个位置的信息素浓度加权随机给出
        # 返回的序列长，是Job数量*工序的数量
        len = self.job_num * self.process_num  # 这里是12
        # 返回的序列，最后这些-1都会被设置成0~job_num-1之间的索引
        ans = [-1 for _ in range(len)]  # 最后生成的是工序序列，代表每个工序是哪个工件
        # 记录每个job使用过的次数，用来防止job被使用超过process_num次
        job_use = [0 for _ in range(self.job_num)]
        # 记录现在还没超过process_num因此可用的job_id，每次满了就将其删除
        job_free = [job_id for job_id in range(self.job_num)]
        # 对于序列的每个位置
        for i in range(len):
            # 把这个位置可用的job的信息素浓度求和
            ph_sum = np.sum(list(map(lambda j: self.topo_phs[i][j], job_free)))
            # 接下来要随机在job_free中取一个job_id
            # 但是不能直接random.choice，要考虑每个job的信息素浓度
            test_val = .0
            rand_ph = random.uniform(0, ph_sum)
            for job_id in job_free:
                test_val += self.topo_phs[i][job_id]
                if rand_ph <= test_val:
                    # 将序列的这个位置设置为job_id，并维护job_use和job_free
                    ans[i] = job_id
                    job_use[job_id] += 1
                    if job_use[job_id] == self.process_num:
                        job_free.remove(job_id)
                    break
        return ans

    def gen_process2machine(self) -> List[List[int]]:
        """
        生成每个Job的每个工序对应的机器索引号矩阵
        :return: 二维int列表，如[0][0]=3表示Job1的p11选择机器m4
        """
        # 要返回的矩阵，共job_num行process_num列，取值0~machine_num-1
        ans = [
            [-1 for _ in range(self.process_num)]
            for _ in range(self.job_num)
        ]
        # 对于每个位置，也是用信息素加权随机出一个machine_id即可
        for job_id in range(self.job_num):
            for process_id in range(self.process_num):
                # 获取该位置的所有可用机器号(times里不为None)
                machine_free = [machine_id for machine_id in range(self.machine_num)
                                if self.times[job_id][process_id][machine_id] is not None]
                # 计算该位置所有可用机器的信息素之和
                ph_sum = np.sum(list(map(lambda m: self.machine_phs[job_id][process_id][m], machine_free)))
                # 还是用随机数的方式选取
                test_val = .0
                rand_ph = random.uniform(0, ph_sum)
                for machine_id in machine_free:
                    test_val += self.machine_phs[job_id][process_id][machine_id]
                    if rand_ph <= test_val:
                        ans[job_id][process_id] = machine_id
                        break
        return ans

    def cal_time(self, topo_jobs: List[int], process2machine: List[List[int]]) -> float:
        """
        给定拓扑序和机器索引号矩阵
        :return: 计算出的总时间花费
        """
        # 记录每个job在拓扑序中出现的次数，以确定是第几个工序
        job_use = [0 for _ in range(self.job_num)]
        # 循环中要不断查询和更新这两张表
        # (1)每个machine上一道工序的结束时间
        machine_end_times = [0 for _ in range(self.machine_num)]
        # (2)每个工件上一道工序的结束时间
        job_end_times = [0 for _ in range(self.job_num)]
        # 对拓扑序中的每个job_id
        for job_id in topo_jobs:
            # 在job_use中取出工序号
            process_id = job_use[job_id]
            # 在process2machine中取出机器号
            machine_id = process2machine[job_id][process_id]
            # 获取max(该job上一工序时间,该machine上一任务完成时间)
            now_start_time = max(job_end_times[job_id], machine_end_times[machine_id])
            # 计算当前结束时间，写入这两个表
            job_end_times[job_id] = machine_end_times[machine_id] = now_start_time + self.times[job_id][process_id][
                machine_id]
            # 维护job_use
            job_use[job_id] += 1
        return max(job_end_times)

    def scheduler_main(self):

        # ----绘图用,横轴和纵轴----
        iter_list = range(self.iteration_num)
        time_list = [0 for _ in iter_list]

        best_topo_jobs = None
        best_process2machine = None

        # 对于每次迭代
        for it in iter_list:
            # 每次迭代寻找最优的<拓扑序,机器分配>方式
            best_time = 9999999

            # 所有蚂蚁各生成一个拓扑序和分配方案，找出其中最好的一组
            # 对于每只蚂蚁
            for ant_id in range(self.ant_num):
                # 生成拓扑序
                topo_jobs = self.gen_topo_jobs()
                # 生成每道工序的分配机器索引号矩阵
                process2machine = self.gen_process2machine()
                # 计算时间
                time = self.cal_time(topo_jobs, process2machine)
                # 如果时间更短，更新最优
                if time < best_time:
                    best_topo_jobs = topo_jobs
                    best_process2machine = process2machine
                    best_time = time
            assert best_topo_jobs is not None and best_process2machine is not None
            # 更新拓扑序信息素浓度表
            for i in range(self.job_num * self.process_num):
                for j in range(self.job_num):
                    if j == best_topo_jobs[i]:
                        self.topo_phs[i][j] *= 1.1
                    else:
                        self.topo_phs[i][j] *= 0.9
            # 更新每个Job的每个工序的信息素浓度表
            for j in range(self.job_num):
                for p in range(self.process_num):
                    for m in range(self.machine_num):
                        if m == best_process2machine[j][p]:
                            self.machine_phs[j][p][m] *= 1.1
                        else:
                            self.machine_phs[j][p][m] *= 0.9
            # 记录时间
            time_list[it] = best_time

        # ----下面是输出内容了----

        # 输出解
        print("\t\t[工序分配给机器的情况]")
        print("\t", end='')
        for machine_id in range(self.machine_num):
            print("\tM{}".format(machine_id + 1), end='')
        print()
        for job_id in range(self.job_num):
            for process_id in range(self.process_num):
                print("p{}{}\t".format(job_id + 1, process_id + 1), end='')
                for machine_id in range(self.machine_num):
                    if machine_id == best_process2machine[job_id][process_id]:
                        print("\t√", end='')
                    else:
                        print("\t-", end='')
                print("")

        print("\n\t\t[工序投给机器的顺序]")
        job_use = [0 for _ in range(self.job_num)]
        for job_id in best_topo_jobs:
            print("p{}{} ".format(job_id + 1, job_use[job_id] + 1), end='')
            job_use[job_id] += 1

        # ----绘图----
        plt.plot(iter_list, time_list)
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.xlabel("迭代轮次")
        plt.ylabel("时间")
        plt.title("柔性作业车间调度-蚁群算法")
        plt.show()


if __name__ == '__main__':
    ac = ACFJSSP()
    ac.scheduler_main()
