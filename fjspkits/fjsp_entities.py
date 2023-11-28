#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/13 20:22
# @Author: ZhaoKe
# @File : fjsp_entities.py
# @Software: PyCharm
# 加工设备
import numpy as np


class Machine(object):
    def __init__(self, machine_id, trans_time_list=None):
        # 机器id
        self.machine_id = machine_id
        # 机器被安排的工序列表
        self.task_list = []
        # 机器的空闲时间
        self.idle_times = []
        # 机器的班次
        self.services = []
        # 机器之间的转运时间列表
        self.trans_time = trans_time_list

    def clear(self):
        self.task_list = []

    def set_trans_time(self, time_list):
        self.trans_time = time_list

    def add_task(self, task):
        self.task_list.append(task)

    def add_service(self, service):
        self.services.append(service)

    def __str__(self):
        return f"Mid: {self.machine_id}: [" + ", ".join([str(i) for i in self.task_list]) + ']'


# 工序
class Task(object):
    def __init__(self, global_index, parent_job, injob_index):
        # 工序所在的作业编号
        self.parent_job = parent_job
        # 工序在作业内的编号
        self.injob_index = injob_index
        # 全体作业所有工序中的编号（无实际意义，仅用于索引）
        self.global_index = global_index
        # 可选机器、偏好、准备时间、执行时间
        self.target_machine = []
        self.machine_prior = []
        self.prepare_time = []
        self.execute_time = []
        # 选定的机器编号与执行时间
        self.selected_machine = None
        self.selected_time = None

        # 下面的属性，通过排程确定
        self.start_time = None
        self.finish_time = None
        # self.can_execute = True if injob_index == 0 else False  # 当前一个工序完成的时候方能执行，把这里设置为True

    def add_alternate_machine(self, machine_id, a_time):
        self.target_machine.append(machine_id)
        self.execute_time.append(a_time)

    def add_four_tuple(self, m_id, m_prior, t_prepare, t_process):
        self.target_machine.append(m_id)
        self.machine_prior.append(m_prior)
        self.prepare_time.append(t_prepare)
        self.execute_time.append(t_process)

    def set_duration(self, st, en):
        self.start_time = st
        self.finish_time = en

    def get_target_machine(self):
        if not self.selected_machine:
            i = np.random.randint(len(self.target_machine))
            self.selected_machine = self.target_machine[i]
            # self.selected_time = self.execute_time[i]
        return self.selected_machine, self.execute_time[self.selected_machine]

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

    def clear_index(self):
        self.cur_index = 0

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
