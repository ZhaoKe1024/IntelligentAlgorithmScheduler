# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2021-08-16 16:11
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def bar_single():
    data = np.loadtxt("chaos-res-matrix/res-0.txt", delimiter=',', encoding="GB2312")
    print("结果的形状", data.shape)
    mean_list = np.mean(data, axis=1)
    std_list = np.std(data, axis=1)
    # 绘制柱状图，设置颜色，设置x轴标签
    name_list = ["CRPSO", "CPSO", "DPSO", "GA", "ACO"]
    plt.bar(range(len(mean_list)), mean_list, fc="blue", tick_label=name_list)
    # plt.bar(range(len(mean_list)), mean_list, color='rgb')
    for x, y in enumerate(mean_list):
        # plt.text(x, 2, y, ha="center", size=7)
        # plt.text(x, y+1, str(round(y*100/sum(mean_list), 1))+'%', ha="center", size=7)
        plt.text(x, y+1, str(round(y, 2)), ha="center", size=7)
    # plt.show()
    plt.savefig('chaos-res-matrix/res-single-bar-0.png', dpi=300, format='png')


def bar_double():
    data = np.loadtxt("doublebar.txt", delimiter=',', encoding="GB2312")
    print(data)
    print("结果的形状", data.shape)
    # mean_list = np.mean(data, axis=1)
    # std_list = np.std(data, axis=1)
    # print(mean_list)
    # print(std_list)
    # # 绘制柱状图，设置颜色，设置x轴标签
    # name_list = ["CRPSO", "CPSO", "DPSO", "GA", "ACO"]
    name_list = ["200", "400", "600", "800", "1000"]
    x = list(range(5))
    total_width, n = 0.6, 3
    width = total_width / n
    plt.bar(x, data[0, :], width=width, label="HEGPSO", fc="red")
    # plt.bar(range(len(mean_list)), mean_list, color='rgb')
    for i in range(5):
        x[i] = x[i] + width
    plt.bar(x, data[1, :], width=width, label="DPSO", tick_label=name_list, fc="green")
    for i in range(5):
        x[i] = x[i] + width
    plt.bar(x, data[2, :], width=width, label="ACO", tick_label=name_list, fc="blue")
    plt.xlabel("task number", fontdict={'size': 18})
    plt.ylabel("score", fontdict={'size': 18})
    plt.xticks(range(5), name_list, size=12)
    plt.yticks(size=12)
    matplotlib.rcParams.update({'font.size': 13})
    plt.legend()
    plt.savefig('double-bar-1.png', dpi=300, format='png')
    # plt.show()


def box_plot(ind):
    # 绘制箱型图
    y0 = np.loadtxt("chaos-res-matrix/res-"+str(ind)+".txt", delimiter=',', encoding="GB2312")
    y0 = np.transpose(y0)
    # r, c = y0.shape
    labels = ["CRPSO", "CPSO", "DPSO", "GA", "ACO"]
    # y0 = 1/y0
    # print(y0.shape)
    # 共5组数据，分别绘制箱型图
    plt.boxplot((y0[70:, 0], y0[:, 1], y0[:, 2], y0[:, 3], y0[:, 4]))
    # plt.plot(range())
    plt.ylabel("fitness", fontdict={'size': 18})
    plt.xticks(range(5), labels, size=16)
    plt.yticks(size=18)
    # plt.plot(range(r), y0[:, 0])
    plt.show()
    # plt.savefig("boxplot" + str(i) + ".png")
    # plt.close(i)


if __name__ == '__main__':
    # bar_single()
    bar_double()
    # box_plot(6)

    # box_plot(0)
    # box_plot(2)
