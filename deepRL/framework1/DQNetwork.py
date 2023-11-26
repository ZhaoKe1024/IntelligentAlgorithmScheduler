#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/20 18:16
# @Author: ZhaoKe
# @File : DQNetwork.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):  # Dueling DQN
    def __init__(self, inputs, outputs, node):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputs, int(node / 2))
        self.fc_value = nn.Linear(int(node / 2), node)
        self.fc_adv = nn.Linear(int(node / 2), node)

        self.value = nn.Linear(node, 1)
        self.adv = nn.Linear(node, outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = F.relu(self.fc_value(x))
        adv = F.relu(self.fc_adv(x))

        value = self.value(value)
        adv = self.adv(adv)

        advAverage = torch.mean(adv, dim=-1, keepdim=True)
        Q = value + adv - advAverage
        return Q


'''
class DQN(nn.Module): #DQN
    def __init__(self, inputs, outputs, node):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputs, node)
        self.fc2 = nn.Linear(node, node)
        self.fc3 = nn.Linear(node, outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
'''
'''
class DQN(nn.Module): #Dueling DQN + LSTM
    def __init__(self, inputs, outputs, node):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputs, int(node/2))
        self.lstm = nn.LSTM(int(node/2), int(node/2), num_layers)
        self.fc_value = nn.Linear(int(node/2), node)
        self.fc_adv = nn.Linear(int(node/2), node)
        self.value = nn.Linear(node, 1)
        self.adv = nn.Linear(node, outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        x = x.unsqueeze(0)
        if(x.dim()==2):
            x = x.unsqueeze(0)
        h0 = torch.zeros(num_layers, x.size(1), int(node/2)).to(device)
        c0 = torch.zeros(num_layers, x.size(1), int(node/2)).to(device) # hidden state와 동일
        x, _ = self.lstm(x, (h0, c0)) #(hn, cn)은 필요 없으므로 받지 않고 _로 처리 
        x = x.squeeze()

        value = F.relu(self.fc_value(x))
        adv = F.relu(self.fc_adv(x))
        value = self.value(value)
        adv = self.adv(adv)

        advAverage = torch.mean(adv, dim=-1, keepdim=True)
        Q = value + adv - advAverage
        return Q
'''