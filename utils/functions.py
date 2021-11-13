# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2021-04-22 11:36


def logistic_function(x, sigma):
    for i in range(len(x)):
        x[i] = sigma * x[i] * (1-x[i])
    return x

