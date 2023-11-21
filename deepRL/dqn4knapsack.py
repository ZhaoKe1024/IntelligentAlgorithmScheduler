#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/20 18:38
# @Author: ZhaoKe
# @File : dqn4knapsack.py
# @Software: PyCharm
import matplotlib.pyplot as plt
from deepRL.DQNTrainer import DQNTrainer


if __name__ == '__main__':
    trainer = DQNTrainer()
    is_train = False
    if is_train:
        trainer.train()
        # np.save('DQN.npy', score_history)
        plt.figure()
        plt.plot(trainer.score_history)
        plt.xlabel('Episode * 10')
        plt.ylabel('price')
        plt.title('D3QN')
        plt.show()
    else:
        trainer.test("./models/dqn4knapsack")
