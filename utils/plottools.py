# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2021-07-24 16:55
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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


if __name__ == '__main__':
    path = "../imgs/stackPlot/result0dat.csv"
    linePlot(path)
    # stackPlot(path)
