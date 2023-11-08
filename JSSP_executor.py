#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/8 12:38
# @Author: ZhaoKe
# @File : JSSP_executor.py
# @Software: PyCharm
from utils.jsp_reader import JSPData
from appkits.jsp_model import JSSP_Optimizer

if __name__ == '__main__':
    data = JSPData("./datasets/fjsp_sets/brandimarte_mk01.txt")
    optimizer = JSSP_Optimizer(data=data)
