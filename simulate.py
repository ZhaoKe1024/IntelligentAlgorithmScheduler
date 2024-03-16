#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/16 17:50
# @Author: ZhaoKe
# @File : simulate.py
# @Software: PyCharm
from schedulers.DPSOTaskScheduling import DPSO
from schedulers.GAScheduler import GAScheduler
from schedulers.SAScheduler import SAScheduler
from eautils.Entities import VM, Cloudlet


def main(use_data, use_algorithm):
    vm_list_path = f"./datasets/cloud_schedule/vm_list_{use_data}.txt"
    task_list_path = f"./datasets/cloud_schedule/task_list_{use_data}.txt"
    vm_params = ["vm_id", "cpu_supply", "cpu_velocity", "mem_supply", "mem_capacity",
                 "hd_supply", "hd_capacity", "bw_supply", "bw_capacity"]
    # lets_params = ["cpu_demand", "mem_demand", "hd_demand", "bw_demand"]
    vm_list = []
    cloudlet_list = []
    with open(vm_list_path, 'r') as vm_file:
        line = vm_file.readline()
        while line:
            parts = line.strip().split(',')
            params = {}
            for i, key in enumerate(vm_params):
                params[key] = float(parts[i])
            vm_list.append(VM(**params))
            line = vm_file.readline()
    with open(task_list_path, 'r') as cl_file:
        line = cl_file.readline()
        while line:
            parts = line.split(',')
            cloudlet_list.append(Cloudlet(cpu_demand=float(parts[0]),
                                          mem_demand=float(parts[1]),
                                          hd_demand=float(parts[2]),
                                          bw_demand=float(parts[3])))
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
    elif use_algorithm == "SA":
        scheduler = SAScheduler(cloudlets=cloudlet_list, vms=vm_list, population_number=500,  times=500)
        res = scheduler.execute()
        i = 0
        for _ in scheduler.best_way:
            print("任务:", i, " 放置到机器", scheduler.vms[scheduler.best_way[i]].id, "上执行")
            i += 1
        # plt.plot(range(sa.times), data)  # 正常应该是2.7左右
        # # plt.savefig('imgr2/BPOScheduler-0.95_2_2--vmax5-popu100-iter200-w095-cg2-cl2.png', dpi=300,
        # #             format='png')  # bbox_inches="tight"解决X轴时间两个字不被保存的问题
        # plt.show()
    elif use_algorithm == "DPSO":
        # 速度最快
        scheduler = DPSO(cloudlets=cloudlet_list, vms=vm_list, times=150)
        scheduler.schedule()


def run():
    use_algorithm = "GA"
    use_data = 1
    main(use_data, use_algorithm)


if __name__ == '__main__':
    run()
