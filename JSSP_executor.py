#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/8 12:38
# @Author: ZhaoKe
# @File : JSSP_executor.py
# @Software: PyCharm
import numpy as np


# 加工设备
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


# 工序
class Task(object):
    def __init__(self, global_index, parent_job):
        self.parent_job = parent_job
        self.global_index = global_index
        # 可选机器和时间
        self.target_machine = []
        self.execute_time = []
        self.selected_machine = None

    def add_alternate_machine(self, machine_id, a_time):
        self.target_machine.append(machine_id)
        self.execute_time.append(a_time)

    def get_target_machine(self):
        if not self.selected_machine:
            self.selected_machine = np.random.randint(len(self.target_machine))
        return self.target_machine[self.selected_machine], self.execute_time[self.selected_machine]

    def __str__(self):
        return f"Task{self.global_index}"


# 工件
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
    # print(len(res))
    while len(jobs_tmp) > 0:
        for i in range(len(jobs_tmp)):
            # print(len(jobs_tmp))
            select_job_index = np.random.randint(len(jobs_tmp))
            # print(select_job_index)
            if jobs_tmp[select_job_index].is_finished():
                jobs_tmp.pop(select_job_index)
            else:
                first_task_this_job = jobs_tmp[select_job_index].give_task_to_machine()
                target_machine, _ = first_task_this_job.get_target_machine()
                # print("machine id:", target_machine)
                res[target_machine].add_task(first_task_this_job)
    return res


def calculate_sum_load(machines):
    res = []
    for machine in machines:
        time_load = 0
        for task in machine.task_list:
            _, e_time = task.get_target_machine()
            time_load += e_time
        res.append(time_load)
    return res


def run():
    with open("./datasets/fjsp_sets/brandimarte_mk01.txt") as fin:
        first_line = fin.readline().strip().split(' ')
        job_num = int(first_line[0])
        machine_num = int(first_line[1])

        job_id = 0
        task_cnt = 0
        line = fin.readline()
        jobs = [Job(i) for i in range(job_num)]
        while line:
            parts = [int(s) for s in line.strip().split(' ')]
            part_index = 1
            while part_index < len(parts):
                tmp_task = Task(task_cnt, job_id)
                task_cnt += 1
                for i in range(part_index + 1, part_index + 2 * parts[part_index] + 1, 2):
                    tmp_task.add_alternate_machine(parts[i]-1, parts[i + 1])
                part_index = part_index + 2 * parts[part_index] + 1
                jobs[job_id].add_task(tmp_task)
            job_id += 1
            line = fin.readline()

    # 因为一个工件的多个工序之间有先后之分，需要以有序的部分做一个矩阵，序号就是列，先求出最大工序数
    # print("最大工序数：", max_sequence_num)

    for j in jobs:
        print(j)

    solution = generate_new_solution(jobs, machine_num)
    for m in solution:
        print(m)
    # print(str(solution))
    print(calculate_sum_load(solution))


if __name__ == '__main__':
    run()
