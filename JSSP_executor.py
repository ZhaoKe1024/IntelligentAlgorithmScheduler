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

#10个问题的最优解区间
IDEAL_TIME = {'MK01': (36, 42), 'MK02': (24, 32), 'MK03': (204, 211), 'MK04': (48, 81), 'MK05': (168, 186), 'MK06': (33, 86), 'MK07': (133, 157), 'MK08': (523, 523), 'MK09': (299, 369), 'MK10': (165, 296)}
BEST_TIME = {'MK01': 40, 'MK02': 26, 'MK03': 204, 'MK04': 60, 'MK05': 171, 'MK06': 57, 'MK07': 139, 'MK08': 523, 'MK09': 307, 'MK10': (165, 296)}

"""
import os

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from fjspkits.fjsp_utils import read_Data_from_file
from fjspkits.FJSP_GAModel import Genetic4FJSP
from utils.plottools import plot_gantt


def run():
    """通过MK01算例的文本获取数据
    jobs: list<Job>类型
    machine_num: 解向量(List<Machine>)的长度
    task_num: 全部工序数，暂时没啥用
    """
    jobs, machine_num, task_num = read_Data_from_file("./datasets/fjsp_sets/brandimarte_mk03.txt")

    ga4fjsp = Genetic4FJSP(jobs=jobs, machine_num=machine_num, task_num=task_num)
    best_gene, results = ga4fjsp.schedule()
    if not os.path.exists("fjspkits/results/mk03"):
        os.mkdir("fjspkits/results/mk03")
    output_prefix = "fjspkits/results/mk03/t" + time.strftime("%Y%m%d%H%M", time.localtime())
    print(results)
    np.savetxt(output_prefix + "_itervalues.txt", results, fmt='%.18e', delimiter=',', newline='\n')

    plt.figure(0)
    plt.plot(results, c='black')
    plt.xlabel("step")
    plt.ylabel("fitness")
    plt.grid()
    plt.savefig(output_prefix + "_iterplot.png", dpi=300, format='png')
    plt.close()

    print("---------------Optimal!-----------------")
    res_in = open(output_prefix + "_planning.txt", 'w')
    makespan = 0.0
    for m in best_gene.get_machines():
        print(m)
    for machine in best_gene.get_machines():
        for task in machine.task_list:
            makespan = makespan if makespan > task.finish_time else task.finish_time
            print(f"Task({task.parent_job}-{task.injob_index})" + f"[{task.start_time},{task.finish_time}]", end='||')
            res_in.write(f"Task({task.parent_job}-{task.injob_index})" + f"[{task.start_time},{task.finish_time}]||")
        print()
        res_in.write('\n')
    # print(f"最大Task数: {self.task_num}")
    res_in.close()

    print(f"最短完工时间：{makespan}")

    # 迄今为止跑出的mk01的最优解为50
    with open(output_prefix + "_minmakespan.txt", 'w', encoding="utf_8") as fin:
        fin.write(f"最短完工时间：{makespan}")
    # ---------------------------Gantt Plot---------------------------------
    # 根据machines得到一个pandas用于绘图
    data_dict = {"Task": {}, "Machine": {}, "Job": {}, "start_num": {}, "end_num": {}, "days_start_to_end": {}}
    for idx, machine in enumerate(best_gene.get_machines()):
        for task in machine.task_list:
            # 修改了这个地方的机器编号，因为我发现有时候甘特图和结果对不上，看来是Task的selected_machine有误，没有正确赋值，还需要检查
            data_dict["Machine"][task.global_index] = "M" + str(idx)
            data_dict["Task"][task.global_index] = f"Task[{task.parent_job}-{task.injob_index}]"
            data_dict["Job"][task.global_index] = "Job" + str(task.parent_job)
            data_dict["start_num"][task.global_index] = task.start_time
            data_dict["end_num"][task.global_index] = task.finish_time
            data_dict["days_start_to_end"][task.global_index] = task.selected_time
    df = pd.DataFrame(data_dict)
    plot_gantt(df, machine_num, fname=output_prefix + "_gantt.png")


if __name__ == '__main__':
    run()
    # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # print(time.strftime("%Y%m%d%H%M", time.localtime()))
