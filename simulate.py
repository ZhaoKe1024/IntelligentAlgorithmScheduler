#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/16 17:50
# @Author: ZhaoKe
# @File : simulate.py
# @Software: PyCharm
from schedulers.GAScheduler import GAScheduler


def main(args):
    ga_schedule = GAScheduler()
    ga_schedule.execute()

def run():
    pass


if __name__ == '__main__':
    run()