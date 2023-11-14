# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2021-07-24 16:55
"""
重要：甘特图绘制 gantt_plot()

"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
from pandas import Timestamp


def linePlot(path):
    data = pd.read_csv(path, delimiter=',', index_col=None, header=None)
    data.drop(data.columns[5], axis=1, inplace=True)
    print(data)
    plt.figure(0)
    labels = ["GA", "ACO", "CPSO", "DPSO", "CRPSO"]
    colors = ['red', 'purple', 'orange', 'green', 'blue']
    x = [1, 2, 3, 4, 5]
    y1 = data.iloc[:, 0]
    y2 = data.iloc[:, 1]
    y3 = data.iloc[:, 2]
    y4 = data.iloc[:, 3]
    y5 = data.iloc[:, 4]
    plt.plot(x, y1, colors[0])
    plt.plot(x, y2, colors[1])
    plt.plot(x, y3, colors[2])
    plt.plot(x, y4, colors[3])
    plt.plot(x, y5, colors[4])
    plt.legend(labels)
    plt.show()


def stackPlot(path):
    data = pd.read_csv(path, delimiter=',', index_col=None, header=None)
    data.drop(data.columns[5], axis=1, inplace=True)
    print(data)
    plt.figure(0)
    labels = ["GA", "ACO", "CPSO", "DPSO", "CRPSO"]
    x = [1, 2, 3, 4, 5]
    y1 = data.iloc[:, 0]
    y2 = data.iloc[:, 1]
    y3 = data.iloc[:, 2]
    y4 = data.iloc[:, 3]
    y5 = data.iloc[:, 4]
    plt.stackplot(x, y1, y2, y3, y4, y5)
    plt.show()


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
        plt.text(x, y + 1, str(round(y, 2)), ha="center", size=7)
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
    y0 = np.loadtxt("chaos-res-matrix/res-" + str(ind) + ".txt", delimiter=',', encoding="GB2312")
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


"""
reference: 
  [1] https://github.com/sgzqc/wechat/blob/main/20211212/gantt_v4.py
  [2] https://blog.csdn.net/weixin_43840701/article/details/107129595
input: 
"""


def plot_gantt(df, machine_num):
    rgb_number = ["#FF0000", "#FFD700", "#FFFF00", "#7CFC00", "#008000", "#00FFFF",
                  "#0000FF", "#6A5ACD", "#800080", "#696969", "#800000", "#D3D3D3"]
    def color(row):
        c_dict = {}
        for i, rgb in enumerate(rgb_number):
            c_dict["Job" + str(i)] = rgb
        return c_dict[row['Job']]

    df['color'] = df.apply(color, axis=1)

    # fig, ax = plt.subplots(1, figsize=(16, 6))
    fig, (ax, ax1) = plt.subplots(2, figsize=(16, 6), gridspec_kw={'height_ratios': [6, 1]})
    # ax.barh(df.Task, df.current_num, left=df.start_num, color=df.color)
    ax.barh(df.Machine, df.days_start_to_end, left=df.start_num, color=df.color, alpha=0.7)
    ax.set_yticks(range(machine_num), ["Machine" + str(i) for i in range(machine_num)])

    # texts
    for idx, row in df.iterrows():
        # ax.text(row.end_num + 0.1, idx, f"{int(row.Completion * 100)}%", va='center', alpha=0.8)
        ax.text(row.start_num + 0.5, row.Machine, row.Task, fontsize=10, va='center', ha='right', alpha=0.8)

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
    legend_elements = []
    for i, rgb in enumerate(rgb_number):
        legend_elements.append(Patch(facecolor=rgb, label=f"Job{i}"))
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
    # path = "../imgs/stackPlot/result0dat.csv"
    # linePlot(path)
    # stackPlot(path)
    # gantt()
    pass
