#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/8 12:38
# @Author: ZhaoKe
# @File : JSSP_executor.py
# @Software: PyCharm
# @version: 0.0.1
"""
这是正在开发中的一个FJSP柔性车间调度模型的遗传算法框架，的最初版本，一个文件包含全部

一个车间有m个Machine，有n个job，每个job有多个task，每个task有多个可以选择的机器，执行时间各不相同。
每个job的多个task之间有顺序约束，不同job之间没有依赖关系。
目标：把所有task分配到m个Machine上，满足job的顺序约束，最小化执行时间。

This is the initial version of the genetic algorithm framework for an FJSP flexible workshop scheduling model under development,
with one file containing all.
A workshop has m machines, n jobs, and each job has multiple tasks. Each task has multiple machines to choose from, and the execution time varies.
There are order constraints between multiple tasks of each job, and there is no dependency relationship between different jobs.
Goal: Allocate all tasks to m machines, meet job order constraints, and minimize execution time.
"""
import copy

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


def generate_new_solution(jobs, machine_num, solution1=None, solution2=None, mode=None):
    res = [Machine(i) for i in range(machine_num)]
    jobs_tmp = jobs.copy()
    # print(len(res))
    if mode == 0:
        # 随机初始化
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
    elif mode == 1:
        # 交叉
        pass
    elif mode == 2:
        # 变异  改变一个随机的job的随机task的machine选择
        # 有bug！会不会因此导致目标任务的工序乱掉？
        print_msg = ""
        i = np.random.randint(len(jobs))
        print_msg += f"select job[{i}]"
        rand_job = jobs[i]
        i = np.random.randint(len(rand_job.task_list))
        print_msg += f" task[{i}], "
        rand_task = copy.copy(rand_job.task_list[i])
        print_msg += f"and its available machines are {rand_task.target_machine}"
        print_msg += f"from machine[{rand_task.selected_machine}] to"

        origin_machine = rand_task.selected_machine
        origin_index = 0
        for j, j_task in enumerate(solution1[origin_machine].task_list):
            if j_task.global_index == rand_task.global_index:
                origin_index = j
        solution1[origin_machine].task_list.pop(origin_index)

        rand_task.get_rand_machine()
        print_msg += f" [{rand_task.selected_machine}], "
        solution1[rand_task.selected_machine].task_list.append(rand_task)

        print(print_msg)
        # print(solution1)
        res = solution1
    elif mode == 3:
        # 片段倒置  选择一个机器，shuffle其中的执行顺序
        # 有bug！会导致目标机器上所有任务的工序乱掉
        pass
    elif mode == 4:
        # 片段互换位置

        pass
    return res


# 选择操作：轮盘赌，概率和其适应度成正比，适应度越好，留下的机会越大，最优的一个为1。
def select():
    pass


# 计算适应度，并返回对齐的结果
def calculate_sum_load(machines, return_align_result=False, job_num=10):
    """
    Since the calculated solution only focuses on which tasks are executed on each machine,
    and has not been aligned according to the process time,
    time must be aligned first before calculating fitness.
    :param job_num:
    :param machines:
    :param return_align_result:
    :return:
    """
    finished = [False for _ in range(len(machines))]  # 是否全部执行完毕
    job_task_index_memory = [0 for _ in range(job_num)]  # 每个job当前执行到哪个task了
    job_end_times_memory = [0 for _ in range(job_num)]  # 记录job当前task的结束时间
    machine_task_index_memory = [0 for _ in range(len(machines))]
    # 遍历机器，遍历task list，记录任务结束时间（这样的方法前提是任务之间没有依赖冲突，如何检测有无乱序呢？）
    while not all(finished):
        for i, machine in enumerate(machines):
            if finished[i]:
                continue
            # time_load = 0
            # print(machine)
            for j in range(machine_task_index_memory[i], len(machine.task_list)):
                cur_task = machine.task_list[j]  # 只读变量，不用于被赋值
                # 当前task可以执行的条件：job执行到的工序，就是当前task的工序号
                if cur_task.injob_index == job_task_index_memory[cur_task.parent_job]:
                    # print(f"current: job[{cur_task.parent_job}] Task[{cur_task.injob_index}]")
                    j_machine, j_time = cur_task.get_target_machine()  # 获取当前task的所在机器和所需时间
                    start_t = None
                    end_t = None
                    if j == 0:
                        if cur_task.injob_index == 0:
                            start_t = 0
                            end_t = j_time
                            # print(f"allocated time 0: job[{cur_task.parent_job}] Task[{cur_task.injob_index}] start:{0} ,end:{j_time}")
                        else:
                            start_t = job_end_times_memory[cur_task.parent_job]
                            end_t = job_end_times_memory[cur_task.parent_job] + j_time
                    else:
                        # 设置当前task开始时间 max(同工件上一个工序的结束时间，同机器前一个task结束时间)
                        # 和结束时间
                        # 考虑一下没有task_list[j-1]的情况
                        if cur_task.injob_index == 0:
                            # print(f"in this machine, last task is :", machine.task_list[j - 1], "its info is", machine.task_list[j - 1].start_time, machine.task_list[j - 1].finish_time)
                            start_t = machine.task_list[j - 1].finish_time
                            end_t = start_t + j_time
                            # print(f"allocated time 1: job[{cur_task.parent_job}] Task[{cur_task.injob_index}] start:{start_t} ,end{end_t}")
                        else:
                            # print(f"injob last task endtime:{job_end_times_memory[cur_task.parent_job]}, inmachine last task endtime:{machine.task_list[j - 1].finish_time}, its index {j - 1}")

                            # print(f"in this machine, last task is :", machine.task_list[j - 1], "its info is", machine.task_list[j - 1].start_time, machine.task_list[j - 1].finish_time)
                            start_t = max(machine.task_list[j - 1].finish_time, job_end_times_memory[cur_task.parent_job])
                            end_t = start_t + j_time
                            # print(f"allocated time 2: job[{cur_task.parent_job}] Task[{cur_task.injob_index}] start:{start_t} ,end{end_t}")
                    machine.task_list[j].start_time = start_t
                    machine.task_list[j].finish_time = end_t
                    # 表示当前job的下一个task可以执行了，不需要判断越界，因为machine已经约束
                    job_task_index_memory[cur_task.parent_job] += 1  # 当前job的下一个task能够执行了
                    # job_task_index_memory[cur_task.selected_machine] += 1  # 当前machine的下一个task能执行了
                    # 下一个task是否能执行？时间如何allocate？
                    job_end_times_memory[cur_task.parent_job] = machine.task_list[j].finish_time
                    machine_task_index_memory[i] += 1
                    # print("job task index memory:\n", job_task_index_memory)
                    # print("job end times memory:\n", job_end_times_memory)
                    # print("machine task index memory:\n", machine_task_index_memory)
                    # print("-------")
                else:
                    break
            if machine_task_index_memory[i] == len(machine.task_list):
                finished[i] = True

    res = []  # Total time executed on each machine
    for machine in machines:
        res.append(machine.task_list[-1].finish_time)
    return res, machines


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
            injob_index = 0
            while part_index < len(parts):
                tmp_task = Task(task_cnt, job_id, injob_index)
                injob_index += 1
                task_cnt += 1
                for i in range(part_index + 1, part_index + 2 * parts[part_index] + 1, 2):
                    tmp_task.add_alternate_machine(parts[i] - 1, parts[i + 1])
                part_index = part_index + 2 * parts[part_index] + 1
                jobs[job_id].add_task(tmp_task)
            job_id += 1
            line = fin.readline()

    # 因为一个工件的多个工序之间有先后之分，需要以有序的部分做一个矩阵，序号就是列，先求出最大工序数
    # print("最大工序数：", max_sequence_num)

    for j in jobs:
        print(j)
    print("----------------------------")
    # 得到的是[job1, job2, ..., jobn]
    # jobi=[task1, task2, ..., taskn]
    # task=[mid, time]
    # 初始化，还有优化空间，尽可能产生一个好的初始解
    solution = generate_new_solution(jobs=jobs, machine_num=machine_num, mode=0)
    for m in solution:
        print(m)
    # print(str(solution))
    print("---------------Key!-----------------")
    res, aligned_machines = calculate_sum_load(solution)
    print("每个机器的完工时间：", res)
    print(f"最短完工时间：{max(res)}")
    for machine in aligned_machines:
        for task in machine.task_list:
            print(f"[{task.start_time},{task.finish_time}]", end='|')
        print()
    # mutation_solution1 = generate_new_solution(jobs=jobs, machine_num=machine_num, solution1=solution, mode=2)
    # for m in mutation_solution1:
    #     print(m)
    # # 适应度函数，时间越小越好
    # res = calculate_sum_load(mutation_solution1)
    # print("每个机器的完工时间：", res)
    # print(f"最短完工时间：{max(res)}")


if __name__ == '__main__':
    run()
