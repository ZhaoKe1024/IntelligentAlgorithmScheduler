#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/13 20:21
# @Author: ZhaoKe
# @File : FJSP_GAModel.py
# @Software: PyCharm
"""

"""
import random
from copy import copy
import time
import numpy as np
import matplotlib.pyplot as plt
from fjspkits.fjsp_struct import Solution, SolutionSortedList
from fjspkits.fjsp_entities import Machine
from fjspkits.fjsp_utils import read_Data_from_file, calculate_exetime_load


class Genetic4FJSP(object):
    def __init__(self, data_file):
        # inputs
        self.jobs, self.machine_num, self.task_num = read_Data_from_file(data_file)
        self.job_num = len(self.jobs)
        # hyperparameters
        self.population_number = 50
        self.iter_steps = 100
        # 自己设置的概率，没有依据
        self.c_times_per_step = 20
        self.p_times_per_step = 40
        self.cp = 0.2  # 0.5(1 2) 0.25(3-5) 0.2(6-12) 0.15(13-18) else 0.1
        self.mp = 0.1  # 1/2 of cp
        self.max_select_p = 0.55
        # variables
        self.genes = SolutionSortedList()
        self.best_gene = None

    def schedule(self):
        for j in self.jobs:
            print(j)
        print("============initialize solution 0===============")
        for i in range(self.population_number):
            self.genes.add_solution(Solution(self.__generate_init_solution(), self.job_num))
        self.genes.update_fitness()
        self.best_gene = self.genes.solutions[0]
        results = [self.best_gene.get_fitness()]

        # 遗传算法迭代
        for t in range(1, self.iter_steps):
            print(f"---->steps {t}/{self.iter_steps}----")
            # --选择--交叉--变异--产生新解
            # 每个step交叉20次，选择40个新解里面最好的一个保留
            b_gene_c1, b_gene_c2, b_gene_m = Solution(None, self.job_num), Solution(None, self.job_num), Solution(None, self.job_num)
            for _ in range(self.c_times_per_step):
                # 想法：交叉算子不考虑精英策略了吗？
                gene_c1, gene_c2 = self.CrossoverPOX(self.genes.get_rand_solution().get_machines(),
                                                     self.genes.get_rand_solution().get_machines())
                if gene_c1.get_fitness() > b_gene_c1.get_fitness():
                    b_gene_c1 = gene_c1
                if gene_c2.get_fitness() > b_gene_c2.get_fitness():
                    b_gene_c2 = gene_c2
            for _ in range(self.p_times_per_step):
                gene_m = self.mutation(self.genes.get_rand_solution().get_machines())
                if gene_m.get_fitness() > b_gene_m:
                    b_gene_m = gene_m
            # 更新适应度
            self.genes.update_fitness()
            # 自然选择，替换种群
            selected_genes = self.natural_selection()

            # 添加 变异、交叉、选择的新解
            for i in range(len(selected_genes)+3):
                self.genes.solutions.pop()
            for ge in selected_genes:
                self.genes.add_solution(ge)
            self.genes.add_solution(b_gene_c1)
            self.genes.add_solution(b_gene_c2)
            self.genes.add_solution(b_gene_m)
            # current best
            self.best_gene = self.genes.solutions[0]
            results.append(self.best_gene.get_fitness())

        output_prefix = "./results/t"+time.strftime("%Y%m%d%H%M", time.localtime())
        print(results)
        np.savetxt(output_prefix+"_itervalues.txt", results, fmt='%.18e', delimiter=',', newline='\n')

        plt.figure(0)
        plt.plot(results, c='black')
        plt.xlabel("step")
        plt.ylabel("fitness")
        plt.grid()
        plt.savefig(output_prefix+"_iterplot.txt", dpi=300, format='png')
        plt.close()

        for m in self.best_gene:
            print(m)
        print("---------------Key!-----------------")
        res, aligned_machines = calculate_exetime_load(self.best_gene, job_num=len(self.jobs))
        print("每个机器的完工时间：", res)
        print(f"最短完工时间：{max(res)}")

        with open(output_prefix+"_minmakespan.txt", 'w') as fin:
            fin.write(f"最短完工时间：{max(res)}")
        res_in = open(output_prefix+"_planning.txt")
        for m in self.best_gene:
            print(m)
        for machine in aligned_machines:
            for task in machine.task_list:
                print(f"Task({task.parent_job}-{task.injob_index})"+f"[{task.start_time},{task.finish_time}]", end='||')
                res_in.write(f"Task({task.parent_job}-{task.injob_index})"+f"[{task.start_time},{task.finish_time}]||")
            print()
            res_in.write('\n')
        print(f"最大Task数: {self.task_num}")
        res_in.close()
        # ---------------------------Gantt Plot---------------------------------
        # # 根据machines得到一个pandas用于绘图
        # data_dict = {"Task": {}, "Machine": {}, "Job": {}, "start_num": {}, "end_num": {}, "days_start_to_end": {}}
        # for machine in aligned_machines:
        #     for task in machine.task_list:
        #         data_dict["Machine"][task.global_index] = "M" + str(task.selected_machine)
        #         data_dict["Task"][task.global_index] = "Task" + str(task.global_index)
        #         data_dict["Job"][task.global_index] = "Job" + str(task.parent_job)
        #         data_dict["start_num"][task.global_index] = task.start_time
        #         data_dict["end_num"][task.global_index] = task.finish_time
        #         data_dict["days_start_to_end"][task.global_index] = task.selected_time
        # df = pd.DataFrame(data_dict)
        # plot_gantt(df, self.machine_num)

    def __generate_init_solution(self):
        """贪心算法初始化，让时间更紧凑（负载均衡并不好用，因为有前后约束，总会有时间浪费，利用率很难提高）"""
        res = [Machine(i) for i in range(self.machine_num)]
        end_time_machines = [0 for _ in range(self.machine_num)]
        jobs_tmp = self.jobs.copy()
        while len(jobs_tmp) > 0:
            for i in range(len(jobs_tmp)):
                # print(len(jobs_tmp))
                select_job_index = np.random.randint(len(jobs_tmp))
                # print(select_job_index)
                if jobs_tmp[select_job_index].is_finished():
                    jobs_tmp.pop(select_job_index)
                else:
                    first_task_this_job = jobs_tmp[select_job_index].give_task_to_machine()
                    # 在可选机器中，选一个时间结束最早的机器；或者另一种策略：在可选机器中，选一个用时最少的机器
                    target_machines = first_task_this_job.target_machine  # list of optional machines
                    target_m_idx = target_machines[0]
                    target_end_time = end_time_machines[target_m_idx]
                    m_index = 0
                    for j, m_idx in enumerate(target_machines):
                        if target_end_time > end_time_machines[m_idx]:
                            target_m_idx = m_idx
                            target_end_time = end_time_machines[m_idx]
                            m_index = j
                    exe_times = first_task_this_job.execute_time
                    end_time_machines[target_m_idx] += exe_times[m_index]
                    first_task_this_job.selected_machine = target_m_idx
                    first_task_this_job.selected_time = exe_times[m_index]
                    res[target_m_idx].add_task(first_task_this_job)
                    # 修改工件下的工序的选择机器
                    self.jobs[first_task_this_job.parent_job].task_list[
                        first_task_this_job.injob_index] = first_task_this_job
                    # print(f"to machine[{target_m_idx}]", end_time_machines)
        return res

    def CrossoverPOX(self, s1, s2):
        """ POX crossover, Precedence Operation Crossover
        首先明确，解的形式是一个链式矩阵，行数等于机器数，每一个机器上是一个工作序列。
        两个解的Crossover操作是，解的对应每一行进行交叉
        每一行的交叉的对象是两个序列
        交叉就按照POX的方式即可、
        reference: 张超勇,饶运清,刘向军等.基于POX交叉的遗传算法求解Job-Shop调度问题[J].中国机械工程,2004(23):83-87.
        :param s1:
        :param s2:
        :return:
        """
        # First, randomly divide the job list into two parts
        # 设置一个概率，不是一定要交叉
        row_num = len(s1)
        new_s1, new_s2 = copy(s1), copy(s2)
        for m_idx in range(row_num):
            if random.random() < self.cp:
                new_s1_mi, new_s2_mi = self.__pox(s1[m_idx], s2[m_idx])
                new_s1[m_idx] = new_s1_mi
                new_s2[m_idx] = new_s2_mi
        return Solution(new_s1, job_num=self.job_num), Solution(new_s2, self.job_num)

    def __pox(self, s1_rowi, s2_rowi):
        job_num = len(self.jobs)

        random_list_index = list(range(job_num))
        random.shuffle(random_list_index)  # 打乱job号
        divide_pos = random.randint(0, job_num)  # 把乱序的Jobs集合分成两部分J1和J2
        new_s1_row = copy(s1_rowi)
        new_s2_row = copy(s2_rowi)
        # 找到两个原gene的J2部分， J1部分保留， J2保留顺序交叉
        allocate_s1 = []  # 保留J2部分的序列索引，交叉用
        allocate_s2 = []
        for idx, task in enumerate(s1_rowi):
            if task.parent_job in random_list_index[divide_pos:]:
                allocate_s1.append(idx)
        for idx, task in enumerate(s2_rowi):
            if task.parent_job in random_list_index[divide_pos:]:
                allocate_s2.append(idx)
        # 交叉操作用双指针方法一遍过
        point1, point2 = 0, 0
        while point1 < len(allocate_s1):
            if point2 < len(allocate_s2):
                new_s2_row[allocate_s2[point2]] = s1_rowi[allocate_s1[point1]]
            else:
                new_s2_row.append(s1_rowi[allocate_s1[point1]])
            point2 += 1
            point1 += 1
        point2, point1 = 0, 0
        while point2 < len(allocate_s2):
            if point1 < len(allocate_s1):
                new_s1_row[allocate_s1[point1]] = s2_rowi[allocate_s2[point2]]
            else:
                new_s1_row.append(s2_rowi[allocate_s2[point2]])
            point1 += 1
            point2 += 1
        return new_s1_row, new_s2_row

    def mutation(self, s):
        """根据文献，变异有 交换变异、插入变异、逆转变异等
        这里复现一下单点自交换编译：我自己把它定义为，任选一个机器，把其中的工序调换一下
        用双指针的方法随机调换同一个机器中的两个Job子任务"""
        machine_num = len(s)
        tar_index = random.randint(0, machine_num)
        tar_machine = s[tar_index]
        jobs_list = []
        for task in tar_machine:
            if task.parent_job not in jobs_list:
                jobs_list.append(task.parent_job)
        if len(jobs_list) > 1:
            j1, j2 = jobs_list[-2], jobs_list[-1]
            # 找到索引为j1, j2的任务所在位置
            j1_index_list = []
            j2_index_list = []
            for idx, task in enumerate(tar_machine):
                if task.parent_job == j1:
                    j1_index_list.append(idx)
                elif task.parent_job == j2:
                    j2_index_list.append(idx)
            new_tar_m = copy(tar_machine)
            point, point1, point2 = 0, 0, 0
            while point1 < len(j1_index_list) and point2 < len(j2_index_list):
                se = random.randint(0, 2)
                point = min(j1_index_list[point1], j2_index_list[point2])
                if se == 0:
                    new_tar_m[point] = j1_index_list[point1]
                    point1 += 1
                else:
                    new_tar_m[point] = j2_index_list[point2]
                    point2 += 1
            while point1 < len(j1_index_list):
                point = min(j1_index_list[point1], j2_index_list[point2])
                new_tar_m[point] = j1_index_list[point1]
                point1 += 1
            while point2 < len(j2_index_list):
                point = min(j1_index_list[point1], j2_index_list[point2])
                new_tar_m[point] = j2_index_list[point2]
                point2 += 1
            new_s = copy(s)
            new_s[tar_index] = new_tar_m
            return Solution(new_s, self.job_num)
        else:
            return Solution(s, self.job_num)

    # 选择操作：轮盘赌，概率和其适应度成正比，适应度越好，留下的机会越大，最优的一个为1。
    # 精英保留策略
    def natural_selection(self):
        """
        当前 step的population内最优的直接保留，但是根据适应度计算阈值必定会让最优留下，所以没事
        但是会不会导致留下的精英太多了，限制了搜索范围，导致局部最优解太明显，这是个很大的问题，思考一下。
        有办法了，可以设置阈值最大为0.5或者0.4
        剩下的按照fitness排序随机选择
        :return:
        """
        res = []
        for so in self.genes.solutions:
            tmp_threshold = self.max_select_p*so.get_fitness()
            if random.random() < tmp_threshold:
                res.append(so)
        # local_best = self.genes.solutions[-1]
        return res

    def check_toposort(self, solution) -> bool:
        """验证一个解是否符合拓扑序，只需要依次验证每一个machine上的task list，在各自job内是否有序即可"""
        for machine in solution:
            job_ids = [-1 for _ in range(self.jobs)]
            for task in machine.task_list:
                if job_ids[task.parent_job] == -1:
                    job_ids[task.parent_job] = task.injob_index
                else:
                    if job_ids[task.parent_job] > task.injob_index:
                        return False
                    else:
                        job_ids[task.parent_job] = task.injob_index
        return True
