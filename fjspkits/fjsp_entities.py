#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/13 20:22
# @Author: ZhaoKe
# @File : fjsp_entities.py
# @Software: PyCharm
# 加工设备
import numpy as np


class Machine(object):
    def __init__(self, machine_id):
        self.machine_id = machine_id
        self.task_list = []
        self.idle_times = []

    def clear(self):
        self.task_list = []

    def add_task(self, task):
        self.task_list.append(task)

    def __str__(self):
        return f"Mid: {self.machine_id}: [" + ", ".join([str(i) for i in self.task_list]) + ']'


# 工序
class Task(object):
    def __init__(self, global_index, parent_job, injob_index):
        self.parent_job = parent_job
        self.injob_index = injob_index
        self.global_index = global_index
        # 可选机器和时间
        self.target_machine = []
        self.execute_time = []
        self.selected_machine = None
        self.selected_time = None

        # 下面的属性，用于计算适应度用
        self.start_time = None
        self.finish_time = None
        # self.can_execute = True if injob_index == 0 else False  # 当前一个工序完成的时候方能执行，把这里设置为True

    def add_alternate_machine(self, machine_id, a_time):
        self.target_machine.append(machine_id)
        self.execute_time.append(a_time)

    def get_target_machine(self):
        if not self.selected_machine:
            i = np.random.randint(len(self.target_machine))
            self.selected_machine = self.target_machine[i]
            self.selected_time = self.execute_time[i]
        return self.selected_machine, self.selected_time

    def get_rand_machine(self):
        i = np.random.randint(len(self.target_machine))
        self.selected_machine = self.target_machine[i]
        self.selected_time = self.execute_time[i]
        return self.target_machine[i], self.execute_time[i]

    def __str__(self):
        # return f"Task{self.global_index}"
        return f"Task-{self.parent_job}-{self.injob_index}-{self.selected_time}"


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
