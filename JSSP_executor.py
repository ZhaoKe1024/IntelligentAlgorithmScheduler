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
# import time
from fjspkits.fjsp_utils import read_Data_from_file
from fjspkits.FJSP_GAModel import Genetic4FJSP


def run():
    """通过MK01算例的文本获取数据
    jobs: list<Job>类型
    machine_num: 解向量(List<Machine>)的长度
    task_num: 全部工序数，暂时没啥用
    """
    jobs, machine_num, task_num = read_Data_from_file("./datasets/fjsp_sets/brandimarte_mk01.txt")

    ga4fjsp = Genetic4FJSP(jobs=jobs, machine_num=machine_num, task_num=task_num)
    ga4fjsp.schedule()


if __name__ == '__main__':
    run()
    # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # print(time.strftime("%Y%m%d%H%M", time.localtime()))
