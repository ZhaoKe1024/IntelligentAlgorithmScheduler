#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/13 20:22
# @Author: ZhaoKe
# @File : fjsp_utils.py
# @Software: PyCharm
import copy
import numpy as np
from fjspkits.fjsp_entities import Machine, Job, Task


def read_Data_from_file(file_path):
    with open(file_path) as fin:
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
    return jobs, machine_num, task_cnt


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


# 计算适应度，并返回对齐的结果
def calculate_exetime_load(machines, job_num=10):
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
                            start_t = max(machine.task_list[j - 1].finish_time,
                                          job_end_times_memory[cur_task.parent_job])
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
