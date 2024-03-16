#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/8 8:42
# @Author: ZhaoKe
# @File : jsp_reader.py
# @Software: PyCharm
"""
Data Structure for Flexible Job-Shop Scheduling Problem.
The difficulty of this program lies in the Design of the Data Structure.
"""
from abc import ABC
import numpy as np


class Data(ABC):
    def __init__(self):
        self.total_number_jobs = 0
        self.total_number_machines = 0
        self.total_number_tasks = 0  # 文件第一列的和，所有工序数， Task就是工序
        self.max_tasks_per_job = 0  # 所有工件中，最多的工序数
        self.machine_speeds = None
        self.jobs = []
        self.job_task_index_matrix = None
        self.task_processing_times_matrix = None
        self.usable_machines_matrix = None
        self.sequence_dependency_matrix = None

    def get_job(self, job_id):
        return self.jobs[job_id]

    def get_runtime(self, job_id, task_id, machine):
        return self.task_processing_times_matrix[self.job_task_index_matrix[job_id, task_id], machine]

    def __str__(self):
        result = "----------------\n" \
                 f"total jobs = {self.total_number_jobs}\n" \
                 f"total tasks = {self.total_number_tasks}\n" \
                 f"total machines = {self.total_number_machines}\n" \
                 f"max tasks for a job = {self.max_tasks_per_job}\n" \
                 f"tasks:\n" \
                 f"[jobId, taskId, sequence, usable_machines, pieces]\n"

        for job in self.jobs:
            for task in job.tasks:
                result += str(task) + '\n'

        if self.sequence_dependency_matrix is not None:
            result += f"sequence_dependency_matrix: {self.sequence_dependency_matrix.shape}\n\n" \
                      f"{self.sequence_dependency_matrix}\n\n"

        if self.job_task_index_matrix is not None:
            result += f"dependency_matrix_index_encoding: {self.job_task_index_matrix.shape}\n\n" \
                      f"{self.job_task_index_matrix}\n\n"

        if self.usable_machines_matrix is not None:
            result += f"usable_machines_matrix: {self.usable_machines_matrix.shape}\n\n" \
                      f"{self.usable_machines_matrix}\n\n"

        if self.task_processing_times_matrix is not None:
            result += f"task_processing_times: {self.task_processing_times_matrix.shape}\n\n" \
                      f"{self.task_processing_times_matrix}\n\n"

        if self.machine_speeds is not None:
            result += f"machine_speeds: {self.machine_speeds.shape}\n\n" \
                      f"{self.machine_speeds}"

        return result


class SpreadsheetData(Data):
    """
    JSSP instance data class for spreadsheet data.

    :type seq_dep_matrix: Path | str | Dataframe
    :param seq_dep_matrix: path to the csv or xlsx file or a Dataframe containing the sequence dependency setup times

    :type machine_speeds: Path | str | Dataframe
    :param machine_speeds: path to the csv or xlsx file or a Dataframe containing all of the machine speeds

    :type job_tasks: Path | str | Dataframe
    :param job_tasks: path to the csv or xlsx file or a Dataframe containing all of the job-tasks

    :returns: None
    """
    def __init__(self):
        """
        Initializes all of the static data from the csv files.

        :type seq_dep_matrix: Path | str | Dataframe
        :param seq_dep_matrix: path to the csv or xlsx file or a Dataframe containing the sequence dependency setup times.
        If this param is None, then it is assumed that all setup times are 0 minutes.

        :type machine_speeds: Path | str | Dataframe
        :param machine_speeds: path to the csv or xlsx file or a Dataframe containing all of the machine speeds

        :type job_tasks: Path | str | Dataframe
        :param job_tasks: path to the csv or xlsx file or a Dataframe containing all of the job-tasks

        :returns: None
        """
        super().__init__()


class JSPData(Data):
    def __init__(self, row_list):
        super().__init__()
        with open(row_list, 'r') as fin:
            lines = [line.strip() for line in fin.readlines() if line]
            first_line = lines[0].split(' ')  # 订单数 设备数 每道工序的可用设备数
            self.total_number_jobs = int(first_line[0])
            self.total_number_machines = int(first_line[1])

            self.total_number_tasks = 0  # 文件第一列的和，所有工序数， Task就是工序
            self.max_tasks_per_job = 0  # 所有工件中，最多的工序数
            # 每一行：工序数 后面1+
            for line in lines[1:]:
                line = [int(s) for s in line.split(' ')]
                num_tasks = line[0]
                self.total_number_tasks += num_tasks
                self.max_tasks_per_job = max(num_tasks, self.max_tasks_per_job)

            # 工序i在机器j上的执行时间value

            self.task_processing_times_matrix = np.full((self.total_number_tasks, self.total_number_machines), -1, dtype=np.float32)
            self.sequence_dependency_matrix = np.zeros((self.total_number_tasks, self.total_number_tasks),
                                                       dtype=np.intc)
            self.usable_machines_matrix = np.empty((self.total_number_tasks, self.total_number_machines),
                                                   dtype=np.intc)
            self.job_task_index_matrix = np.full((self.total_number_jobs, self.max_tasks_per_job), -1,
                                                 dtype=np.intc)

            task_index = 0
            for row_id, line in enumerate(lines[1:]):
                self.jobs.append(Job(job_id=row_id))
                parts = [int(s) for s in line.split(' ')]
                task_id = 0
                sequence = 0
                index = 1
                while index < len(parts):
                    num_usable_machine = parts[index]
                    usable_machines = []
                    for j in range(index+1, index+num_usable_machine*2, 2):
                        machine = parts[j] - 1
                        run_time = parts[j+1]
                        usable_machines.append(machine)
                        self.task_processing_times_matrix[task_id, machine] = run_time

                    self.jobs[row_id].tasks.append(Task(row_id, task_id, sequence, usable_machines, -1))
                    self.job_task_index_matrix[row_id, task_id] = task_index
                    self.usable_machines_matrix[task_index] = np.resize(np.array(usable_machines, dtype=np.intc), self.total_number_machines)
                    task_index += 1
                    sequence += 1
                    index += 2 * num_usable_machine + 1
                    task_id += 1

                self.jobs[row_id].set_max_sequence(sequence-1)


class Task(object):
    def __init__(self, job_id, task_id, sequence, usable_machines, pieces):
        self.job_id = job_id
        self.task_id = task_id
        self.sequence = sequence
        self.usable_machines = usable_machines
        self.pieces = pieces

    def __eq__(self, other):
        return self.job_id == other.job_id \
            and self.task_id == other.task_id \
            and self.sequence == other.sequence \
            and np.array_equal(self.usable_machines, other.usable_machines)  # note pieces are omitted

    def __str__(self):
        return f"[{self.job_id}, " \
               f"{self.task_id}, " \
               f"{self.sequence}, " \
               f"{self.usable_machines}, " \
               f"{self.pieces}]"


class Job(object):
    def __init__(self, job_id):
        self.job_id = job_id
        self.tasks = []
        self.max_sequence = 0

    def set_max_sequence(self, max_sequence):
        self.max_sequence = max_sequence

    def __eq__(self, other):
        return self.job_id == other.job_id \
            and self.max_sequence == other.max_sequence \
            and self.tasks == other.tasks


if __name__ == '__main__':
    jsp1 = JSPData("../datasets/fjsp_sets/brandimarte_mk01.txt")
    print(jsp1.total_number_jobs)
    print(jsp1.total_number_machines)
    print(jsp1.total_number_tasks)
