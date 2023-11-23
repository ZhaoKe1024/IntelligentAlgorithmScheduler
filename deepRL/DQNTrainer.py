#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/20 18:20
# @Author: ZhaoKe
# @File : DQNTrainer.py
# @Software: PyCharm
import math
import os.path
import random
import numpy as np
from collections import deque
import torch
import torch.optim as optim
import torch.nn.functional as F
from BaseTrainer import BaseTrainer
from deepRL.env_knapsack import KnapsackEnv
from deepRL.DQNModules import ReplayMemory, PrioritizedMemory
from deepRL.DQNetwork import DQN

# https://github.com/dayoung08/lab/blob/f5e2ec66c5f2a6e1965428b06eff9dcc7c7a595d/DRL/01knapsack-value_based(DQN%2CDDQN%2CD3QN).ipynb#L209


def setup_seed():
    SEED = 1004
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


class DQNTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.env = KnapsackEnv()
        self.score_history = []
        self.EPISODES = 1000
        self.EPS_START = 0.9  # 0.7527517282492975  # 0.9
        self.EPS_END = 0.05  # 0.21341763105951902  # 0.05
        self.EPS_DECAY = 200  # 292.577488609503  # 200
        self.GAMMA = 0.8  # 0.6816764048146292  # 0.8 # discount factor
        self.LR = 0.0011  # learning rate
        setup_seed()

        self.BATCH_SIZE = 512  # batch size
        self.TARGET_UPDATE = 50
        self.node = 32
        self.TD_ERROR_EPSILON = 0.0001  #
        self.c_constant = 192
        self.num_layers = 1

        self.number_times_action_selected = np.zeros(self.env.item_num)

        self.replay_memory = ReplayMemory(10000)
        self.td_error_memory = PrioritizedMemory(10000)
        self.steps_done = 0

        self.train_from_zero = True
        self.best_price = -1

    def __setup_model(self):
        n_states = self.env.get_state_space()
        n_actions = self.env.get_action_space()
        self.item_num = self.env.get_item_num()

        self.policy_net = DQN(n_states, n_actions, self.node).to(self.device)
        self.target_net = DQN(n_states, n_actions, self.node).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), self.LR)

    def learn(self, e):
        if len(self.replay_memory) < self.BATCH_SIZE:
            return
        batch = None
        if e == 0:
            batch = self.replay_memory.sample(self.BATCH_SIZE)
        else:
            batch = self.td_error_memory.sample(self.BATCH_SIZE, self.replay_memory)
        states, actions, rewards, next_states = zip(*batch)
        # Tensor list
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)

        self.policy_net.train()
        self.target_net.train()

        current_q = self.policy_net(states.to(self.device)).gather(1, actions.to(self.device))

        a = self.policy_net(states.to(self.device)).data.max(-1)[1].unsqueeze(1)
        max_next_q = self.target_net(next_states.to(self.device)).gather(1, a)

        ''' #DQN on policy_net
            max_next_q = policy_net(next_states).max(1)[0].unsqueeze(1) # get max_next_q at poicy_net
            '''
        ''' DQN on target_net
        max_next_q = target_net(next_states).max(1)[0].unsqueeze(1) # get max_next_q at targety_net
        '''
        expected_q = rewards.to(self.device) + (self.GAMMA * max_next_q)  # rewards + future value
        loss = F.mse_loss(current_q, expected_q)
        loss_value = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.policy_net.eval()
        self.target_net.eval()
        return loss_value

    def train(self, save_model_path="./models/dqn4knapsack/"):
        if not self.policy_net:
            self.__setup_model()
        if not self.train_from_zero:
            self.__load_checkpoint(save_model_path)
        for e in range(1, self.EPISODES + 1):
            state = self.env.reset()
            steps = 0
            price = 0
            while True:
                state = torch.FloatTensor(state)
                action = self.choose_action(state, e, 1)  # 1: ubc; 0: e_greedy
                action = torch.tensor([action])

                next_state, reward, done = self.env.step(action)
                reward = torch.tensor([reward], dtype=torch.float32)
                next_state = torch.FloatTensor(next_state)

                self.replay_memory.push(state, action, reward, next_state)
                self.td_error_memory.push(0)
                loss = self.learn(e)

                state = next_state
                steps += 1

                if done:
                    if loss is None:
                        loss = -100
                    print("Episode:{0} step: {1} price: {2} reward: {3:0.3f} loss: {4:0.3f}".format(e, steps, price, reward.item(), loss))
                    if e % 10 == 0:
                        self.score_history.append(price)
                    if e % 200 == 0:
                        temp_price = self.evaluate()
                        if temp_price > self.best_price:
                            self.best_price =temp_price
                            best_model = True
                        else:
                            best_model = False
                        self.__save_checkpoint(save_model_path, e, best_model=best_model)
                    self.update_td_error_memory()
                    break
                else:
                    price += self.env.prices[action]

            if e % self.TARGET_UPDATE == 0:
                """"""
                self.target_net.load_state_dict(self.policy_net.state_dict())
                print("Policy_net to Target_net")

    def evaluate(self):
        if not self.policy_net:
            self.__setup_model()
        temp_env = KnapsackEnv()
        state = temp_env.reset()
        steps = 0
        price = 0
        while True:
            state = torch.FloatTensor(state)
            m = np.array(state, dtype=bool)
            action = np.ma.array(self.policy_net(state.to(self.device)).data.cpu(), mask=m).argmax().item()
            action = torch.tensor([action])

            next_state, reward, done = temp_env.step(action)
            reward = torch.tensor([reward], dtype=torch.float32)
            next_state = torch.FloatTensor(next_state)
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
                price += temp_env.prices[action]
        return price

    def test(self, resume_model_path):
        if not self.policy_net:
            self.__setup_model()
        self.__load_checkpoint(resume_model_path)
        state = self.env.reset()
        steps = 0
        price = 0
        selection = []
        while True:
            state = torch.FloatTensor(state)
            m = np.array(state, dtype=bool)
            action = np.ma.array(self.policy_net(state.to(self.device)).data.cpu(), mask=m).argmax().item()
            action = torch.tensor([action])

            next_state, reward, done = self.env.step(action)
            reward = torch.tensor([reward], dtype=torch.float32)
            next_state = torch.FloatTensor(next_state)
            if state.tolist() == next_state.tolist():
                print("The error occurs - price: {0}".format(price))
                break
            state = next_state
            steps += 1

            if done:
                # selection = []
                # for i in range(len(state)):
                #     if state[i] > 0.0001:
                #         selection.append(i)
                for s in selection:
                    print(f"[{s},{self.env.weights[s]},{self.env.prices[s]}]", end=', ')
                print()
                print("step: {0} price: {1}".format(steps, price))
                break
            else:
                print(state.tolist(), "->", next_state.tolist())
                selection.append(action)
                price += self.env.prices[action]
        print("done")

    def choose_action(self, state, e, method):
        select_action = -100
        m = np.array(state, dtype=bool)
        if method == 0:
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
                -1. * self.steps_done / self.EPS_DECAY)
            if random.random() > eps_threshold:
                with torch.no_grad():
                    select_action = np.ma.array(self.policy_net(state.to(self.device)).data.cpu(),
                                                mask=m).argmax().item()
            else:
                action_list = [i for i in range(self.env.item_num) if m[i] == False]
                select_action = random.choice(action_list)
        else:
            select_action = np.ma.array(self.policy_net(state.to(self.device)).data.cpu() + self.c_constant * np.sqrt(
                np.log((self.steps_done + 1) + 0.1) / (self.number_times_action_selected + 0.1)),
                                        mask=m).argmax().item()

        self.steps_done += 1
        self.number_times_action_selected[select_action] += 1
        return select_action

    def update_td_error_memory(self):
        batch = self.replay_memory.memory
        states, actions, rewards, next_states = zip(*batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)

        current_q = self.policy_net(states.to(self.device)).gather(1, actions.to(self.device))
        a = self.policy_net(states.to(self.device)).data.max(-1)[1].unsqueeze(1)
        max_next_q = self.target_net(next_states.to(self.device)).gather(1, a)
        expected_q = rewards.to(self.device) + (self.GAMMA * max_next_q)

        td_errors = expected_q - current_q
        self.td_error_memory.memory = deque(td_errors.detach().cpu().squeeze().numpy().tolist())
