#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/13 20:21
# @Author: ZhaoKe
# @File : FJSP_GAModel.py
# @Software: PyCharm
import numpy as np
import pandas as pd

from fjspkits.fjsp_entities import Machine
from fjspkits.fjsp_utils import read_Data_from_file, calculate_sum_load
from utils.plottools import plot_gantt


class Genetic4FJSP(object):
    def __init__(self, data_file):
        self.jobs, self.machine_num, self.task_num = read_Data_from_file(data_file)

    def schedule(self):
        for j in self.jobs:
            print(j)
        s0 = self.__init_solution()
        for m in s0:
            print(m)
        print("---------------Key!-----------------")
        res, aligned_machines = calculate_sum_load(s0, job_num=len(self.jobs))
        print("每个机器的完工时间：", res)
        print(f"最短完工时间：{max(res)}")
        for machine in aligned_machines:
            for task in machine.task_list:
                print(f"[{task.start_time},{task.finish_time}]", end='|')
            print()
        print(f"最大Task数: {self.task_num}")
        # 根据machines得到一个pandas用于绘图
        data_dict = {"Task": {}, "Machine": {}, "Job": {}, "start_num": {}, "end_num": {}, "days_start_to_end": {}}
        for machine in aligned_machines:
            for task in machine.task_list:
                data_dict["Machine"][task.global_index] = "M" + str(task.selected_machine)
                data_dict["Task"][task.global_index] = "Task" + str(task.global_index)
                data_dict["Job"][task.global_index] = "Job" + str(task.parent_job)
                data_dict["start_num"][task.global_index] = task.start_time
                data_dict["end_num"][task.global_index] = task.finish_time
                data_dict["days_start_to_end"][task.global_index] = task.selected_time
        df = pd.DataFrame(data_dict)
        plot_gantt(df, self.machine_num)

    def __init_solution(self):
        # 随机初始化
        res = [Machine(i) for i in range(self.machine_num)]
        end_time_machines = [0 for _ in range(self.machine_num)]
        jobs_tmp = self.jobs.copy()
        while len(jobs_tmp) > 0:
            for i in range(len(jobs_tmp)):
                # print(len(jobs_tmp))
                select_job_index = np.random.randint(len(jobs_tmp))
                # print(select_job_index)
                if jobs_tmp[select_job_index].is_finished():
                    jobs_tmp.pop(select_job_index)
                else:
                    first_task_this_job = jobs_tmp[select_job_index].give_task_to_machine()
                    # 选择一个负载最低的，或者结束时间最短的（贪心算法初始化）
                    target_machines = first_task_this_job.target_machine
                    target_m_idx = target_machines[0]
                    target_end_time = end_time_machines[target_m_idx]
                    m_index = 0
                    for m_idx in target_machines:
                        if target_end_time > end_time_machines[m_idx]:
                            target_m_idx = m_idx
                            target_end_time = end_time_machines[m_idx]
                            m_index += 1
                    exe_times = first_task_this_job.execute_time
                    end_time_machines[target_m_idx] += exe_times[m_index]
                    res[target_m_idx].add_task(first_task_this_job)
                    # print(f"to machine[{target_m_idx}]", end_time_machines)
        return res

    def crossover(self, s1, s2):
        pass

    def mutation(self, s):
        pass

    # 选择操作：轮盘赌，概率和其适应度成正比，适应度越好，留下的机会越大，最优的一个为1。
    def natural_selection(self):
        pass
