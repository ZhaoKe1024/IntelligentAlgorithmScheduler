# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2021-03-20 9:12
import numpy as np
from matplotlib import pyplot as plt
from utils.dataExamples import get_data_r2n4c18


# def print2csv(ind: int):
#     # 打印节点信息
#     data = get_data(ind)["nodes"]
#     print("#第", ind, "组实验")
#     print("节点资源信息")
#     for it in range(len(data)):
#         print(data[it].cpu_supply, ',', data[it].cpu_velocity, ',', data[it].mem_supply, ',',
#               data[it].mem_capacity)
#     # 打印任务信息
#     print("任务请求信息")
#     data = get_data(ind)["cloudlets"]
#     for it in range(len(data)):
#         print(data[it].cpu_demand, ',', data[it].mem_demand)


# def read_plot():
#     data = np.loadtxt("imgr2/result-0409-r2n3c12-1/generation0.txt")
#     L = data.shape[1]
#     ax = plt.figure(0)
#     for y in data:
#         plt.plot(range(L), y)
#     ax.legend(["HPRPSO", "DPSO", "SA", "ACO", "GA"], bbox_to_anchor=(1.01, 0.88), loc=2, borderaxespad=0)
#     plt.subplots_adjust(right=1)
#     plt.savefig('imgr2/test-3.png', dpi=300,
#                 format='png', bbox_inches='tight')  # bbox_inches="tight"解决X轴时间两个字不被保存的问题
#     # plt.show()
#

# 获取不同数据集的不同具体数据
def get_task(task_id: int, data_id: int):
    if task_id == 1:
        return get_data_r2n4c18(data_id)
    # elif task_id == 2:
    #     return get_data_r2n5c20(data_id)
    # elif task_id == 3:
    #     return get_data_r3n4c18(data_id)


def scheduler_test(t_id, eid: int, cycle_id, txt_path):
    # data = get_task(t_id, eid)
    #
    # # 加一个功能，用于将数据存储到txt里面，方便查看
    # # print2csv(eid)
    #
    # print("#第", cycle_id, "次实验")
    # print("节点数量：", len(data["nodes"]), ", 任务数量：", len(data["cloudlets"]))
    # population = 500
    MAX_GEN = 300
    #
    # hprpso = HPRPSO(cloudlets=data["cloudlets"], vms=data["nodes"], population_number=500, times=MAX_GEN)
    # dpso = DPSO(cloudlets=data["cloudlets"], vms=data["nodes"], population_number=population, times=MAX_GEN)
    # # sa = SAScheduler(cloudlets=data["cloudlets"], vms=data["nodes"], population_number=population, times=MAX_GEN)
    # ac = ACScheduler(cloudlets=data["cloudlets"], vms=data["nodes"], population_number=population, times=MAX_GEN)
    # # ga = GAScheduler(cloudlets=data["cloudlets"], vms=data["nodes"], population_number=population, times=MAX_GEN)
    # hprpso_generation = hprpso.exec()
    # dpso_generation = dpso.exec()
    # # sa_generation = sa.sa_main()
    # ac_generation = ac.scheduler_main()
    # print(hprpso_generation, dpso_generation[-1], ac_generation[-1])
    # np.savetxt("doublebar-1.txt", [hprpso_generation[-1], dpso_generation[-1], ac_generation[-1]], newline=",\n")

    ax = plt.figure(cycle_id)
    data = np.loadtxt("imgr2/result-1023-r2n4c18-4/generation4.txt")
    # 固定配
    p1, = plt.plot(range(MAX_GEN), data[0, :], color="red")
    # p2, = plt.plot(range(MAX_GEN), bpso.exec(), color="orange")
    p3 = plt.plot(range(MAX_GEN), data[1, :], color="green")
    p4, = plt.plot(range(MAX_GEN), data[2, :], color="blue")
    # p5, = plt.plot(range(MAX_GEN), sa.scheduler_main(), color="red")
    # ga_generation = ga.ga_main()
    # =====================
    # 绘制折线图
    # =====================
    # p1, = plt.plot(range(MAX_GEN), hprpso_generation)
    # p2, = plt.plot(range(MAX_GEN), dpso_generation)
    # p3, = plt.plot(range(MAX_GEN), ac_generation)
    # # p3, = plt.plot(range(MAX_GEN), sa_generation)
    # # p5, = plt.plot(range(MAX_GEN), ga_generation)

    # data = [hprpso.gbest.solution, dpso.gbest.solution, ac.best_topo]
    # np.savetxt(txt_path+str(eid)+"/result" + str(cycle_id) + ".txt", data, fmt="%d", newline=",\n")
    # generation = [hprpso_generation, dpso_generation, ac_generation]
    # np.savetxt(txt_path+str(eid)+"/generation" + str(cycle_id) + ".txt", generation, fmt="%d", newline="\n")

    ax.legend([p1, p3, p4], ["HPRPSO", "DPSO", "ACO"], bbox_to_anchor=(1.01, 0.88), loc=2, borderaxespad=0)
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    # plt.title("云计算任务调度-负载均衡得分图", fontdict={'size': 18})
    plt.xlabel("generation", fontdict={'size': 18})
    plt.ylabel("fitness", fontdict={'size': 18})
    plt.xticks(size=12)
    plt.yticks(size=12)
    # plt.subplots_adjust(right=1)
    plt.savefig('lineplot.png', dpi=300,
                format='png', bbox_inches='tight')  # bbox_inches="tight"解决X轴时间两个字不被保存的问题
    # plt.savefig(txt_path+str(eid)+'/Scheduler-iter300-' + str(cycle_id) + '.png', dpi=300,
    #             format='png', bbox_inches='tight')  # bbox_inches="tight"解决X轴时间两个字不被保存的问题
    # # plt.show()


if __name__ == '__main__':
    # read_plot()
    path = "imgr2/result-1023-r2n4c18-"
    # # 数据号 文件号 id
    # scheduler_test(1, 2, 0, txt_path=path)
    scheduler_test(1, 3, 1, txt_path=path)
    # scheduler_test(1, 4, 2, txt_path=path)
    # scheduler_test(1, 4, 3, txt_path=path)
    # scheduler_test(1, 4, 4, txt_path=path)
    # scheduler_test(1, 4, 5, txt_path=path)
    # scheduler_test(1, 4, 6, txt_path=path)
    # scheduler_test(1, 4, 7, txt_path=path)
    # scheduler_test(1, 4, 8, txt_path=path)
    # scheduler_test(1, 4, 9, txt_path=path)
