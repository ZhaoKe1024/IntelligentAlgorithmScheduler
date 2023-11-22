#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/22 20:42
# @Author: ZhaoKe
# @File : PPOTrainer.py
# @Software: PyCharm
from collections import deque
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from env_knapsack import KnapsackEnv
from DQNModules import ReplayMemory, PrioritizedMemory
from A2CNetwork import ValueNetwork, PolicyNetwork


class PPOTrainer(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        SEED = 3407
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)

        self.score_history = []  # array to store reward

        # hyperparameters definition
        # EPISODES = 50000 #수렴하는 때로 맞추기
        self.EPISODES = 10000  # 수렴하는 때로 맞추기 LSTM용
        self.GAMMA = 0.6816764048146292  # 0.8 # discount factor

        # V_LR = 0.0030001 # learning rate
        self.V_LR = 0.05  # learning rate LSTM
        self.P_LR = self.V_LR  # learning rate

        self.node = 32
        self.AD_EPSILON = 0.0001  # 오차에 더해줄 바이어스

        # value_loss_coef = 0.5 # critic 손실함수 계산에 사용
        self.clip_param = 0.2  # PPO에 사용
        self.max_grad_norm = 0.1
        self.num_layers = 1

        self.env = KnapsackEnv()
        self.n_states = self.env.get_state_space()
        self.n_actions = self.env.get_action_space()
        self.item_num = self.env.get_item_num()
        self.steps_done = 0

        self.BATCH_SIZE = 10
        self.replay_memory = ReplayMemory(self.BATCH_SIZE * 5)
        self.advantage_memory = PrioritizedMemory(self.BATCH_SIZE * 5)

    def __setup_model(self):
        self.v_net = ValueNetwork(inputs=self.n_states, outputs=self.n_actions,
                                  node=self.node, num_layers=self.num_layers, device=self.device).to(self.device)
        self.p_net = PolicyNetwork(inputs=self.n_states, outputs=self.n_actions,
                                   node=self.node, num_layers=self.num_layers, device=self.device).to(self.device)
        self.v_optimizer = optim.Adam(self.v_net.parameters(), self.V_LR)
        self.p_optimizer = optim.Adam(self.p_net.parameters(), self.P_LR)

    def __save_checkpoint(self, save_model_path, epoch_id, best_model=False):
        policy_state_dict = self.p_net.state_dict()
        target_state_dict = self.v_net.state_dict()
        os.makedirs(save_model_path, exist_ok=True)
        if best_model:
            policy_model_path = os.path.join(save_model_path, f"policynet_bestmodel.pth").replace('\\', '/')
            target_model_path = os.path.join(save_model_path, f"targetnet_bestmodel.pth").replace('\\', '/')
        else:
            policy_model_path = os.path.join(save_model_path, f"policynet_epoch{epoch_id}.pth").replace('\\', '/')
            target_model_path = os.path.join(save_model_path, f"targetnet_epoch{epoch_id}.pth").replace('\\', '/')
        torch.save(policy_state_dict, policy_model_path)
        torch.save(target_state_dict, target_model_path)
        torch.save(self.optimizer.state_dict(), os.path.join(save_model_path, "optimizer.pth").replace('\\', '/'))

    def __load_checkpoint(self, save_model_path):
        last_epoch = -1
        assert os.path.exists(os.path.join(save_model_path, "optimizer.pth").replace('\\',
                                                                                     '/')), f"{save_model_path}/optimizer.pth not found!"
        assert os.path.exists(os.path.join(save_model_path, "policynet_bestmodel.pth").replace('\\',
                                                                                               '/')), "polictnet_bestmodel.pth not found!"
        assert os.path.exists(os.path.join(save_model_path, "targetnet_bestmodel.pth").replace('\\',
                                                                                               '/')), "targetnet_bestmodel.pth not found!"
        self.policy_net.load_state_dict(
            torch.load(os.path.join(save_model_path, "policynet_bestmodel.pth").replace('\\', '/')))
        self.target_net.load_state_dict(
            torch.load(os.path.join(save_model_path, "targetnet_bestmodel.pth").replace('\\', '/')))
        self.optimizer.load_state_dict(torch.load(os.path.join(save_model_path, "optimizer.pth").replace('\\', '/')))
        self.optimizer.step()
        [self.optimizer.step() for _ in range(last_epoch)]  # 这里存疑，不知道需要多少步骤

    def learn(self, e):
        batch = None
        if e == 0:
            batch = self.replay_memory.sample(self.BATCH_SIZE)
        else:  # 여기도 확률에 따라 일반 샘플링 할지 PER 기반 샘플링 할 지 결정하면 좋을 듯
            batch = self.advantage_memory.sample(self.BATCH_SIZE, self.replay_memory)  # 어드밴티지를 이용해 배치를 추출

        states, actions, rewards, next_states, old_action_log_probs = zip(
            *batch)  # separate rollouts object by element list

        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)
        old_action_log_probs = torch.stack(old_action_log_probs)

        self.v_net.train()
        self.p_net.train()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-10)  #
        for n in range(self.BATCH_SIZE):
            value_targets = rewards.to(self.device).view(-1, 1) + self.GAMMA * self.v_net.forward(next_states.to(self.device)).view(-1, 1)
            values = self.v_net.forward(states.to(self.device))
            advantages = value_targets - values.to(self.device)
            value_loss = advantages.pow(2).mean()  # Critic의 loss
            value_loss_value = value_loss.item()

            self.v_optimizer.zero_grad()
            value_loss.backward()
            # for param in v_net.parameters():
            #    param.grad.data.clamp_(-1, 1)
            nn.utils.clip_grad_norm_(self.v_net.parameters(),
                                     self.max_grad_norm)
            self.v_optimizer.step()

            # policy loss
            actor_output = self.p_net.forward(next_states[n].to(self.device))
            log_probs = F.log_softmax(actor_output, dim=0)  # 행동에 대해 확률을 계산
            action_log_probs = log_probs.gather(0, actions[n].to(self.device))

            adv = float(advantages[n])
            ratio = torch.exp(action_log_probs - old_action_log_probs.to(self.device))
            L1 = ratio * adv
            L2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv
            policy_loss = -torch.min(L1, L2).mean()  # MAX->MIN desent
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
        value_loss = policy_loss = 0
        for e in range(1, self.EPISODES + 1):
            state = self.env.reset()
            steps = 0
            price = 0
            while True:
                state = torch.FloatTensor(state)  # tensorize state
                action, action_log_prob = self.choose_action(state)
                action = torch.tensor(action, dtype=torch.int64)  # tensorize action
                action_log_prob = torch.FloatTensor(action_log_prob)  # tensorize action_log_prob

                next_state, reward, done = self.env.step(action)
                reward = torch.tensor(reward, dtype=torch.float64)  # tensorize reward
                next_state = torch.FloatTensor(next_state)  # tensorize next_state

                self.replay_memory.push(state, action, reward, next_state, action_log_prob)  # replay_memory experience
                self.advantage_memory.push(0)

                if len(self.replay_memory) == self.replay_memory.memory.maxlen:
                    self.update_advantage_memory()
                    value_loss, policy_loss = self.learn(e)
                    self.replay_memory.memory.clear()
                    self.advantage_memory.memory.clear()

                state = next_state
                steps += 1

                if done:
                    if e % 10 == 0:
                        self.score_history.append(price)
                    break
                else:
                    price += self.env.prices[action]

            # 에피소드 종료
            print("Episode:{0} step: {1} price: {2} reward: {3:0.3f} v_loss: {4:0.3f} p_loss: {5:0.3f}".format(e, steps,
                                                                                                               price,
                                                                                                               reward.item(),
                                                                                                               value_loss,
                                                                                                               policy_loss))

    def test(self):
            pass

    def update_advantage_memory(self):
        batch = self.replay_memory.memory
        states, actions, rewards, next_states, action_log,probs = zip(*batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)

        # value loss
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-10)
        value_targets = rewards.to(self.device).view(-1, 1) + self.GAMMA * self.v_net.forward(next_states.to(self.device)).view(-1, 1)
        values = self.v_net.forward(states.to(self.device))
        advantages = value_targets - values.to(self.device)
        self.advantage_memory = deque(advantages.detach().cpu().squeeze().numpy().tolist())

    def choose_action(self, state):
        with torch.no_grad():
            m = np.array(state,dtype=bool)
            action_targets = np.ma.array(range(self.env.item_num), mask=m).compressed()
            actor_output = np.ma.array(self.p_net(state.to(self.device)).data.cpu(), mask=m).compressed()
            actor_output = torch.FloatTensor(actor_output)
            select_index = F.softmax(actor_output, dim=0).multinomial(num_samples=1)
            log_probs = F.log_softmax(actor_output, dim=0)
            action_log_probs = log_probs[select_index]
            select_action = action_targets[select_index]
            self.steps_done += 1
        return select_action, action_log_probs
