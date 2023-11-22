#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/22 8:26
# @Author: ZhaoKe
# @File : A2CTrainer.py
# @Software: PyCharm
from collections import namedtuple

import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from deepRL.A2CModules import RolloutStorage
from deepRL.A2CNetwork import ValueNetwork, PolicyNetwork
from deepRL.env_knapsack import KnapsackEnv


class A2CTrainer(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.score_history = []
        self.EPISODES = 10000
        self.GAMMA = 0.8
        self.V_LR = 0.0022001
        self.P_LR = self.V_LR
        self.node = 32

        self.value_loss_coef = 0.5  # A2C
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.05
        self.num_layers = 1

        self.steps_done = 0
        self.env = KnapsackEnv()
        self.number_times_action_selected = np.zeros(self.env.item_num)
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
        self.n_states = self.env.get_state_space()
        self.n_actions = self.env.get_action_space()
        self.item_num = self.env.get_item_num()

        self.rollouts = RolloutStorage(self.n_states, self.GAMMA)

    def __setup_model(self):
        self.v_net = ValueNetwork(inputs=self.n_states, outputs=self.n_actions,
                                  node=self.node, num_layers=self.num_layers, device=self.device).to(self.device)
        self.p_net = PolicyNetwork(inputs=self.n_states, outputs=self.n_actions,
                                   node=self.node, num_layers=self.num_layers, device=self.device).to(self.device)
        self.v_optimizer = optim.Adam(self.v_net.parameters(), self.V_LR)
        self.p_optimizer = optim.Adam(self.p_net.parameters(), self.P_LR)

    def choose_action(self, state):
        with torch.no_grad():
            m = np.array(state, dtype=bool)
            action_targets = np.ma.array(range(self.env.item_num), mask=m).compressed()
            actor_output = np.ma.array(self.p_net(state.to(self.device)).data.cpu(), mask=m).compressed()
            actor_output = torch.FloatTensor(actor_output)
            select_index = F.softmax(actor_output, dim=0).multinomial(num_samples=1)
            select_action = action_targets[select_index]
            self.steps_done += 1
        return select_action

    def learn(self):
        states, actions, rewards, next_states = zip(*self.rollouts.memory)  # separate rollouts object by element list

        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)

        self.v_net.train()
        self.p_net.train()

        # value loss
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-10)
        value_targets = rewards.to(self.device).view(-1, 1) + torch.FloatTensor(list(self.rollouts.dis_rewards)).view(-1, 1).to(
            self.device)
        values = self.v_net.forward(states.to(self.device))
        advantages = value_targets - values.to(self.device)
        value_loss = advantages.pow(2).mean()  # Critic的loss计算
        value_loss = value_loss * self.value_loss_coef
        value_loss_value = value_loss.item()

        # 修改合并权重
        self.v_optimizer.zero_grad()
        value_loss.backward()
        # for param in v_net.parameters():
        #    param.grad.data.clamp_(-1, 1)
        nn.utils.clip_grad_norm_(self.v_net.parameters(),
                                 self.max_grad_norm)  # 将斜率限制在max_grad_norma以下，以避免合并权重一次变化太大（修剪）
        self.v_optimizer.step()  # 修改合并权重

        # policy loss
        actor_output = self.p_net.forward(next_states[len(self.rollouts) - 1].to(self.device))
        log_probs = F.log_softmax(actor_output, dim=0)  # 计算action的概率

        action_log_probs = log_probs.gather(0, actions.to(self.device))
        probs = F.softmax(actor_output, dim=0)
        entropy = -(log_probs * probs).sum().mean()
        policy_loss = (action_log_probs.to(self.device) * advantages.detach()).mean()
        policy_loss = -policy_loss - (entropy * self.entropy_coef)
        policy_loss_value = policy_loss.item()

        self.p_optimizer.zero_grad()
        policy_loss.backward()
        # for param in p_net.parameters():
        #    param.grad.data.clamp_(-1, 1)
        nn.utils.clip_grad_norm_(self.p_net.parameters(),
                                 self.max_grad_norm)
        self.p_optimizer.step()

        self.v_net.eval()
        self.p_net.eval()

        return value_loss_value, policy_loss_value

    def train(self):
        self.__setup_model()
        for e in range(1, self.EPISODES + 1):
            state = self.env.reset()
            steps = 0
            price = 0
            while True:
                state = torch.FloatTensor(state)  # tensorize state
                action = self.choose_action(state)
                action = torch.tensor(action, dtype=torch.int64)  # tensorize action

                next_state, reward, done = self.env.step(action)
                reward = torch.tensor(reward, dtype=torch.float64)  # tensorize reward
                next_state = torch.FloatTensor(next_state)  # tensorize next_state

                # 在rollout对象中保存当前步骤的transition
                self.rollouts.push(state, action, reward, next_state)
                state = next_state
                steps += 1

                if done:
                    if e % 10 == 0:
                        self.score_history.append(price)
                    break
                else:
                    price += self.env.prices[action]

            with torch.no_grad():
                next_value = self.v_net(next_state.to(self.device)).detach().cpu()  # 根据state计算value
                self.rollouts.compute_returns(next_value.item())

            value_loss, policy_loss = self.learn()
            self.rollouts.clear()
            # if value_loss == None:
            #    value_loss = -100
            # if policy_loss == None:
            #    policy_loss = -100
            print("Episode:{0} step: {1} price: {2} reward: {3:0.3f} v_loss: {4:0.3f} p_loss: {5:0.3f}".format(e, steps,
                                                                                                               price,
                                                                                                               reward.item(),
                                                                                                               value_loss,
                                                                                                               policy_loss))

    def test(self):
        state = self.env.reset()
        steps = 0
        price = 0
        while True:
            state = torch.FloatTensor(state)  # tensorize state

            m = np.array(state, dtype=bool)  # 마스크
            action_targets = np.ma.array(range(self.env.item_num), mask=m).compressed()
            actor_output = np.ma.array(self.p_net(state.to(self.device)).detach().cpu(), mask=m).compressed()
            actor_output = torch.FloatTensor(actor_output)
            select_index = F.softmax(actor_output, dim=0).argmax()
            action = action_targets[select_index]

            action = torch.tensor(action, dtype=torch.int64)  # tensorize action

            next_state, reward, done = self.env.step(action)  # 1은 아무 값이나 준 거
            reward = torch.tensor(reward, dtype=torch.float64)  # tensorize reward
            next_state = torch.FloatTensor(next_state)  # tensorize next_state

            if state.tolist() == next_state.tolist():
                print("The error occurs - price: {0}".format(price))
                break

            state = next_state
            steps += 1

            if done:
                print("step: {0} price: {1}".format(steps, price))
                break
            else:
                print(state.tolist(), "->", next_state.tolist())
                price += self.env.prices[action]

        print("done")
