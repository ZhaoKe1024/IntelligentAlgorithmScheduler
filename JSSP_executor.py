#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/8 12:38
# @Author: ZhaoKe
# @File : JSSP_executor.py
# @Software: PyCharm
import numpy as np


class Machine(object):
    def __init__(self, machine_id):
        self.machine_id = machine_id
        self.task_list = []

    def clear(self):
        self.task_list = []

    def add_task(self, task):
        self.task_list.append(task)

    def __str__(self):
        return f"Mid: {self.machine_id}: [" + ", ".join([str(i) for i in self.task_list]) + ']'


class Task(object):
    def __init__(self, target_machine, execute_time, parent_job, global_index):
        self.target_machine = target_machine
        self.execute_time = execute_time
        self.parent_job = parent_job
        self.global_index = global_index

    def __str__(self):
        return f"Task{self.global_index}"


class Job(object):
    def __init__(self, job_id):
        self.job_id = job_id
        self.task_list = []
        self.is_executed = []
        self.cur_index = 0

    def add_task(self, task):
        self.task_list.append(task)

    def full_mask(self):
        self.is_executed = [False for _ in range(len(self.task_list))]

    def is_finished(self) -> bool:
        return self.cur_index == len(self.task_list)

    def give_task_to_machine(self):
        res = self.task_list[self.cur_index]
        self.cur_index += 1
        return res

    def __str__(self):
        return f"Jid: {self.job_id}: [" + ", ".join([str(i) for i in self.task_list]) + ']'


def generate_new_solution(jobs, machine_num):
    jobs_tmp = jobs.copy()
    res = [Machine(i) for i in range(machine_num)]
    while len(jobs_tmp) > 0:
        for i in range(len(jobs_tmp)):
            select_job_index = np.random.randint(len(jobs_tmp))
            if jobs_tmp[select_job_index].is_finished():
                jobs_tmp.pop(select_job_index)
            else:
                first_task_this_job = jobs_tmp[select_job_index].give_task_to_machine()
                res[first_task_this_job.target_machine].add_task(first_task_this_job)
    return res


def calculate_sum_load(machines):
    res = []
    for machine in machines:
        time_load = sum([task.execute_time for task in machine.task_list])
        res.append(time_load)
    return res


def run():
    job_num = 4
    machine_num = 4
    job_mask = [1, 1, 1, 2, 2, 3, 3, 4, 4]
    task_machine_map = [1, 4, 1, 4, 2, 4, 2, 3, 3]
    task_machine_time = [2, 5, 8, 7, 6, 11, 13, 1, 5]
    jobs = [Job(i) for i in range(job_num)]
    # 因为一个工件的多个工序之间有先后之分，需要以有序的部分做一个矩阵，序号就是列，先求出最大工序数
    # max_sequence_num = 0
    for j, job_index in enumerate(job_mask):
        jobs[job_index - 1].add_task(Task(task_machine_map[j]-1, task_machine_time[j], job_index - 1, j))
    # print("最大工序数：", max_sequence_num)
    for j in jobs:
        print(j)

    solution = generate_new_solution(jobs, machine_num)
    for m in solution:
        print(m)
    # print(str(solution))
    print(calculate_sum_load(solution))


if __name__ == '__main__':
    # data = JSPData("./datasets/fjsp_sets/brandimarte_mk01.txt")
    # gene = JSPSolutionGenerator(data)
    # new_so = gene.init_solution()
    # print(new_so)
    # optimizer = JSSP_Optimizer(data=data)
    run()
