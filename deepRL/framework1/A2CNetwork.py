#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/22 8:30
# @Author: ZhaoKe
# @File : A2CNetwork.py
# @Software: PyCharm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNetwork(nn.Module):  # 添加了 LSTM
    def __init__(self, inputs, outputs, node, num_layers, device):
        self.num_layers = num_layers
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(inputs, node)
        self.lstm2 = nn.LSTM(node, node, num_layers)
        self.fc3 = nn.Linear(node, 1)
        self.node = node
        self.device = device
        self.num_layers = num_layers

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(0)
        if (x.dim() == 2):
            x = x.unsqueeze(0)

        h0 = torch.zeros(self.num_layers, x.size(1), self.node).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.node).to(self.device)  # 和 hidden state 相同

        x, _ = self.lstm2(x, (h0, c0))  # 不必(hn, cn)
        x = x.squeeze()

        x = self.fc3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, inputs, outputs, node, num_layers, device):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(inputs, node)
        self.lstm2 = nn.LSTM(node, node, num_layers)
        self.fc3 = nn.Linear(node, outputs)
        self.node = node
        self.device = device
        self.num_layers = num_layers

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # input x : (BATCH, LENGTH, INPUT_SIZE) 可以处理各种length
        x = x.unsqueeze(0)
        if (x.dim() == 2):
            x = x.unsqueeze(0)

        h0 = torch.zeros(self.num_layers, x.size(1), self.node).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.node).to(self.device)  # hidden state
        x, _ = self.lstm2(x, (h0, c0))
        x = x.squeeze()

        x = self.fc3(x)
        return x


'''
class ValueNetwork(nn.Module):
    def __init__(self, inputs, outputs, node):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(inputs, node)
        self.fc2 = nn.Linear(node, node)
        self.fc3 = nn.Linear(node, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PolicyNetwork(nn.Module):
    def __init__(self, inputs, outputs, node):
        super(PolicyNetwork, self).__init__()
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
class ActorCritic(nn.Module): #(n_states, n_actions, node) # TwoHeadNetwork... 별로인 것 같다
    def __init__(self, inputs, outputs, node):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(inputs, node)
        self.fc2 = nn.Linear(node, node)
        self.actor = nn.Linear(node, outputs)  # 행동을 결정하는 부분이므로 출력 갯수는 행동의 가짓수
        self.critic = nn.Linear(node, 1)  # 상태가치를 출력하는 부분이므로 출력 갯수는 1개

    def forward(self, x): 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actor_output = self.actor(x)  # 행동 계산
        critic_output = self.critic(x)  # 상태가치 계산
        return critic_output, actor_output
'''
