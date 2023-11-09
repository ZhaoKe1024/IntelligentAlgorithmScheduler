#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/8 12:01
# @Author: ZhaoKe
# @File : fjsp_model.py
# @Software: PyCharm
"""
JSSP Job Shop Scheduling Problem
FJSP Flexible JobShop Scheduling Problem

"""
import random

import numpy as np

from utils.jsp_reader import JSPData, SpreadsheetData


class JSSP_Optimizer(object):
    def __init__(self, data):
        self.data = data
        self.solution_initializer = JSPSolutionGenerator(data=data)


class JSPSolution(object):
    """
    Solution class of JSSP

    """

    def __init__(self, data: JSPData, operation_2d_array):
        if operation_2d_array.shape[0] != data.total_number_tasks:
            raise IncompleteSolutionException(f"Incomplete Solution of size {operation_2d_array.shape[0]}. "
                                              f"Should be {data.total_number_tasks}")
        self.machine_makespans = compute_machine_makespans(operation_2d_array,
                                                           data.task_processing_times_matrix,
                                                           data.sequence_dependency_matrix,
                                                           data.job_task_index_matrix)
        self.makespan = max(self.machine_makespans)
        self.operation_2d_array = operation_2d_array
        self.data = data


class JSPSolutionGenerator(object):
    def __init__(self, data: JSPData):
        self.jsp_data_object = data

    # 初始化一个随机解
    def init_solution(self):
        operation_list = []
        last_task_scheduled_on_machine = [None] * self.jsp_data_object.total_number_machines

        available = {}
        for job in self.jsp_data_object.jobs:
            available[job.job_id] = []
            for task in job.tasks:
                if task.sequence != 0:
                    available[job.job_id].append(task)
        while 0 < len(available):
            get_unstuck = 0
            rand_job_id = random.choice(list(available.keys()))  # 工件
            rand_task = random.choice(available[rand_job_id])  # 工序
            rand_machine = np.random.choice(rand_task.usable_machines)  # 机器
            # 执行时间将从映射表查询

            if isinstance(self.jsp_data_object, SpreadsheetData):
                while last_task_scheduled_on_machine[rand_machine] is not None \
                        and last_task_scheduled_on_machine[rand_machine].job_id == rand_job_id \
                        and last_task_scheduled_on_machine[rand_machine].sequence + 1 < rand_task.get_sequence():

                    rand_job_id = random.choice(list(available.keys()))
                    rand_task = random.choice(available[rand_job_id])
                    rand_machine = np.random.choice(rand_task.get_usable_machines())
                    get_unstuck += 1
                    if get_unstuck > 50:
                        return self.init_solution()  # TODO this is not the best way to do this...


            available[rand_job_id].remove(rand_task)
            if len(available[rand_job_id]) == 0:
                if rand_task.sequence == self.jsp_data_object.jobs[rand_job_id].max_sequence:
                    del available[rand_job_id]
                else:
                    available[rand_job_id] = [t for t in self.jsp_data_object.jobs[rand_job_id].tasks if t.sequence==rand_task.sequence+1]
            last_task_scheduled_on_machine[rand_machine] = rand_task
            operation_list.append([rand_job_id, rand_task.task_id, rand_task.sequence, rand_machine])

        return JSPSolution(self.jsp_data_object, np.array(operation_list, dtype=np.intc))


class InfeasibleSolutionException(Exception):
    pass


class IncompleteSolutionException(Exception):
    pass


def compute_machine_makespans(operation_2d_array, task_processing_times_matrix, sequence_dependency_matrix,
                              job_task_index_matrix):
    num_jobs = sequence_dependency_matrix.shape[0]  # number of tasks
    num_machines = task_processing_times_matrix.shape[1]  # number of machines
    # memory for keeping track of all machine's make span time
    machine_makespan_memory = np.zeros(num_machines)

    # memory for keeping track of all machine's latest job that was processed
    machine_jobs_memory = [-1 for _ in range(num_machines)]
    machine_tasks_memory = [-1 for _ in range(num_machines)]

    job_seq_memory = [0 for _ in range(num_jobs)]
    prev_job_end_memory = [0.0 for _ in range(num_jobs)]
    job_end_memory = [0.0 for _ in range(num_jobs)]

    for row in range(operation_2d_array.shaoe[0]):
        job_id = operation_2d_array[row, 0]
        task_id = operation_2d_array[row, 1]
        sequence = operation_2d_array[row, 2]
        machine = operation_2d_array[row, 3]

        if machine_jobs_memory[machine] != -1:
            cur_task_index = job_task_index_matrix[job_id, task_id]  # what task_index is job i task_id j
            prev_task_index = job_task_index_matrix[machine_jobs_memory[machine], machine_tasks_memory[machine]]
            setup = sequence_dependency_matrix[cur_task_index, prev_task_index]
        else:
            setup = 0
        if setup < 0 or sequence < job_seq_memory[job_id]:
            raise InfeasibleSolutionException()

        if job_seq_memory[job_id] < sequence:
            prev_job_end_memory[job_id] = job_end_memory[job_id]

        if prev_job_end_memory[job_id] <= machine_makespan_memory[machine]:
            wait = 0
        else:
            wait = prev_job_end_memory[job_id] - machine_makespan_memory[machine]

        runtime = task_processing_times_matrix[job_task_index_matrix[job_id, task_id], machine]

        # compute total added time and update memory modules
        machine_makespan_memory[machine] += runtime + wait + setup
        job_end_memory[job_id] = max(machine_makespan_memory[machine], job_end_memory[job_id])
        job_seq_memory[job_id] = sequence
        machine_jobs_memory[machine] = job_id
        machine_tasks_memory[machine] = task_id
    return machine_makespan_memory
