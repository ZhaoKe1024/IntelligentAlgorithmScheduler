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
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from fjspkits.fjsp_entities import Job, Task
from fjspkits.fjsp_utils import generate_new_solution, calculate_sum_load


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
    # 【初始化，还有优化空间，尽可能产生一个好的初始解】
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
    print(f"最大Task数: {task_cnt}")
    # 根据machines得到一个pandas用于绘图
    data_dict = {"Task": {}, "Machine":{},  "Job": {}, "start_num": {}, "end_num": {}, "days_start_to_end": {}}
    for machine in aligned_machines:
        for task in machine.task_list:
            data_dict["Machine"][task.global_index] = "M"+str(task.selected_machine)
            data_dict["Task"][task.global_index] = "Task" + str(task.global_index)
            data_dict["Job"][task.global_index] = "Job" + str(task.parent_job)
            data_dict["start_num"][task.global_index] = task.start_time
            data_dict["end_num"][task.global_index] = task.finish_time
            data_dict["days_start_to_end"][task.global_index] = task.selected_time
    df = pd.DataFrame(data_dict)
    # project start date
    # proj_start = df.Start.min()
    # number of days from project start to task start
    # df['start_num'] = (df.Start - proj_start).dt.days
    # number of days from project start to end of tasks
    # df['end_num'] = (df.End - proj_start).dt.days
    # days between start and end of each task
    # df['days_start_to_end'] = df.end_num - df.start_num
    # df['current_num'] = (df.days_start_to_end * df.Completion)

    def color(row):
        rgb_number = ["#FF0000", "#FFD700", "#FFFF00", "#7CFC00", "#008000", "#00FFFF",
                      "#0000FF", "#6A5ACD", "#800080", "#696969", "#800000", "#D3D3D3"]
        c_dict = {}
        for i, rgb in enumerate(rgb_number):
            c_dict["Job" + str(i)] = rgb
        return c_dict[row['Job']]

    df['color'] = df.apply(color, axis=1)

    # fig, ax = plt.subplots(1, figsize=(16, 6))
    fig, (ax, ax1) = plt.subplots(2, figsize=(16, 6), gridspec_kw={'height_ratios': [6, 1]})
    # ax.barh(df.Task, df.current_num, left=df.start_num, color=df.color)
    ax.barh(df.Machine, df.days_start_to_end, left=df.start_num, color=df.color, alpha=0.7)
    ax.set_yticks(range(machine_num), ["Machine"+str(i) for i in range(machine_num)])

    # texts
    for idx, row in df.iterrows():
        # ax.text(row.end_num + 0.1, idx, f"{int(row.Completion * 100)}%", va='center', alpha=0.8)
        ax.text(row.start_num+0.1, row.Machine, row.Task, fontsize=10, va='center', ha='right', alpha=0.8)

    # grid lines
    ax.set_axisbelow(True)
    ax.xaxis.grid(color='gray', linestyle='dashed', alpha=0.2, which='both')

    # ticks
    xticks = np.arange(0, df.end_num.max() + 1, 3)
    # xticks_labels = pd.date_range(proj_start, end=df.End.max()).strftime("%m/%d")
    xticks_labels = pd.date_range(0, end=df.end_num.max()).strftime("%m/%d")
    xticks_minor = np.arange(0, df.end_num.max() + 1, 1)
    ax.set_xticks(xticks)
    ax.set_xticks(xticks_minor, minor=True)
    # ax.set_xticklabels(xticks_labels[::3])

    # ticks top
    # create a new axis with the same y
    ax_top = ax.twiny()

    # align x axis
    ax.set_xlim(0, df.end_num.max())
    ax_top.set_xlim(0, df.end_num.max())

    # top ticks (markings)
    xticks_top_minor = np.arange(0, df.end_num.max() + 1, 7)
    ax_top.set_xticks(xticks_top_minor, minor=True)
    # top ticks (label)
    xticks_top_major = np.arange(3.5, df.end_num.max() + 1, 7)
    ax_top.set_xticks(xticks_top_major, minor=False)
    # week labels
    xticks_top_labels = [f"Week {i}" for i in np.arange(1, len(xticks_top_major) + 1, 1)]
    ax_top.set_xticklabels(xticks_top_labels, ha='center', minor=False)

    # hide major tick (we only want the label)
    ax_top.tick_params(which='major', color='w')
    # increase minor ticks (to marks the weeks start and end)
    ax_top.tick_params(which='minor', length=8, color='k')

    # remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['top'].set_visible(False)

    ax_top.spines['right'].set_visible(False)
    ax_top.spines['left'].set_visible(False)
    ax_top.spines['top'].set_visible(False)

    plt.suptitle('FJSP Allocate Result Gantt')

    ##### LEGENDS #####
    legend_elements = [Patch(facecolor='#E64646', label='Marketing'),
                       Patch(facecolor='#E69646', label='Finance'),
                       Patch(facecolor='#34D05C', label='Engineering'),
                       Patch(facecolor='#34D0C3', label='Production'),
                       Patch(facecolor='#3475D0', label='IT')]

    ax1.legend(handles=legend_elements, loc='upper center', ncol=5, frameon=False)

    # clean second axis
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.set_xticks([])
    ax1.set_yticks([])

    plt.show()


if __name__ == '__main__':
    run()
