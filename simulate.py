#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/16 17:50
# @Author: ZhaoKe
# @File : simulate.py
# @Software: PyCharm
from schedulers.GAScheduler import GAScheduler
from utils.Entities import VM, Cloudlet


def main(use_data, use_algorithm):
    vm_list_path = f"./datasets/vm_list_{use_data}.txt"
    task_list_path = f"./datasets/task_list_{use_data}.txt"
    vm_list = []
    cloudlet_list = []
    with open(vm_list_path, 'r') as vm_file:
        line = vm_file.readline()
        while line:
            parts = line.split(',')
            vm_list.append(VM(vm_id=int(parts[0]),
                              cpu_supply=float(parts[1]),
                              cpu_velocity=float(parts[2]),
                              mem_supply=float(parts[3]),
                              mem_capacity=float(parts[4])))
            line = vm_file.readline()
    with open(task_list_path, 'r') as cl_file:
        line = cl_file.readline()
        while line:
            parts = line.split(',')
            cloudlet_list.append(Cloudlet(float(parts[0]), float(parts[1]), float(parts[2])))
            line = cl_file.readline()
    scheduler = None
    if use_algorithm == "GA":
        scheduler = GAScheduler(cloudlets=cloudlet_list, vms=vm_list)
        res = scheduler.execute()
        best_gene = scheduler.best_gene
        print("best solution: \n", best_gene.fitness)
        i = 0
        for _ in best_gene.solution:
            print("任务:", i, " 放置到机器", scheduler.vms[best_gene.solution[i]].id, "上执行")
            i += 1
        print(res)


def run():
    use_algorithm = "GA"
    use_data = 0
    main(use_data, use_algorithm)


if __name__ == '__main__':
    run()
