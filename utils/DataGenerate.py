# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2021-04-01 15:12
import numpy as np
import matplotlib.pyplot as plt


def get_sample_data():
    data = []
    with open('cloudlets20r2data.txt', 'r') as f:
        for i in range(20):
            str = f.readline()
            nums = str[10:-5].split(', ')
            # print(nums)
            nums = list(map(float, nums))
            # print(nums)
            data.append(nums)
        f.close()
    # data = np.array(data, )
    # print(data)
    temp = np.sum(data, axis=0)
    NUM_MAX1 = temp[0]
    NUM_MAX2 = temp[1]
    num = 20
    # CPU
    data_mean1 = NUM_MAX1 / num
    print(data_mean1)
    new_data1 = np.random.normal(loc=data_mean1 - 0.0596, scale=0.06, size=num)
    # 内存
    data_mean2 = NUM_MAX2 / num
    print(data_mean2)
    new_data2 = np.random.normal(loc=data_mean2 - 37, scale=47, size=num)
    # # IO
    # data_mean3 = NUM_MAX3 / num
    # print(data_mean3)
    # new_data3 = np.random.normal(loc=data_mean3 - 0.708, scale=0.308, size=num)

    data = np.array([new_data1, new_data2])

    data = data.transpose()
    # print(data)
    for i in data:
        print('Cloudlet(', i[0], ',', i[1], '),')


def generateDataFromSrcData():
    data = []
    with open('cloudlets20r2data.txt', 'r') as f:
        for i in range(20):
            str = f.readline()
            nums = str[10:-5].split(', ')
            # print(nums)
            nums = list(map(float, nums))
            # print(nums)
            data.append(nums)
        f.close()
    data = np.array(data)
    print(data)
    # means = np.mean(data, axis=0)
    # stds = np.std(data, axis=0)
    # new_data = []
    # for i in range(2):
    #     new_data.append(np.random.normal(loc=means[i], scale=stds[i], size=12))
    # new_data = np.array(new_data).transpose()
    #
    for it in data:
        print("Cloudlet(", it[0], ',', it[1], "),")
    # # pass


if __name__ == '__main__':
    get_sample_data()
    # generateDataFromSrcData()
