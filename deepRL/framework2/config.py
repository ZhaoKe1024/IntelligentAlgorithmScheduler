#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/26 16:42
# @Author: ZhaoKe
# @File : config.py
# @Software: PyCharm
import os
import torch
import numpy as np
from typing import List
from torch import Tensor
from multiprocessing import Pipe, Process


class Config:
    def __init__(self, agent_class=None, env_class=None, env_args=None):
        self.num_envs = None
        self.agent_class = agent_class  # agent = agent_class(...)
        self.if_off_policy = self.get_if_off_policy()  # whether off-policy or on-policy of DRL algorithm

        '''Argument of environment'''
        self.env_class = env_class  # env = env_class(**env_args)
        self.env_args = env_args  # env = env_class(**env_args)
        if env_args is None:  # dummy env_args
            env_args = {'env_name': None,
                        'num_envs': 1,
                        'max_step': 12345,
                        'state_dim': None,
                        'action_dim': None,
                        'if_discrete': None, }
        env_args.setdefault('num_envs', 1)  # `num_envs=1` in default in single env.
        env_args.setdefault('max_step', 12345)  # `max_step=12345` in default, which is a large enough value.
        self.env_name = env_args['env_name']  # the name of environment. Be used to set 'cwd'.
        self.num_envs = env_args['num_envs']  # the number of sub envs in vectorized env. `num_envs=1` in single env.
        self.max_step = env_args['max_step']  # the max step number of an episode. 'set as 12345 in default.
        self.state_dim = env_args['state_dim']  # vector dimension (feature number) of state
        self.action_dim = env_args['action_dim']  # vector dimension (feature number) of action
        self.if_discrete = env_args['if_discrete']  # discrete or continuous action space

        '''Arguments for reward shaping'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256

        '''Arguments for training'''
        self.net_dims = (64, 32)  # the middle layer dimension of MLP (MultiLayer Perceptron)
        self.learning_rate = 6e-5  # the learning rate for network updating
        self.clip_grad_norm = 3.0  # 0.1 ~ 4.0, clip the gradient after normalization
        self.state_value_tau = 0  # the tau of normalize for value and state `std = (1-std)*std + tau*std`
        self.soft_update_tau = 5e-3  # 2 ** -8 ~= 5e-3. the tau of soft target update `net = (1-tau)*net + tau*net1`

        if self.if_off_policy:  # off-policy
            self.batch_size = int(64)  # num of transitions sampled from replay buffer.
            self.horizon_len = int(512)  # collect horizon_len step while exploring, then update networks
            self.buffer_size = int(1e6)  # ReplayBuffer size. First in first out for off-policy.
            self.repeat_times = 1.0  # repeatedly update network using ReplayBuffer to keep critic's loss small
            self.if_use_per = False  # use PER (Prioritized Experience Replay) for sparse reward
        else:  # on-policy
            self.batch_size = int(128)  # num of transitions sampled from replay buffer.
            self.horizon_len = int(2048)  # collect horizon_len step while exploring, then update network
            self.buffer_size = None  # ReplayBuffer size. Empty the ReplayBuffer for on-policy.
            self.repeat_times = 8.0  # repeatedly update network using ReplayBuffer to keep critic's loss small
            self.if_use_vtrace = False  # use V-trace + GAE (Generalized Advantage Estimation) for sparse reward
        '''Arguments for device'''
        self.gpu_id = int(0)  # `int` means the ID of single GPU, -1 means CPU
        # For Windows: self.num_workers = 0
        self.num_workers = 0  # rollout workers number pre GPU (adjust it to get high GPU usage)
        self.num_threads = 8  # cpu_num for pytorch, `torch.set_num_threads(self.num_threads)`
        self.random_seed = 0  # initialize random seed in self.init_before_training()
        self.learner_gpus = 0  # `int` means the ID of single GPU, -1 means CPU

        '''Arguments for evaluate'''
        self.cwd = None  # current working directory to save model. None means set automatically
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = np.inf  # break training if 'total_step > break_step'
        self.break_score = np.inf  # break training if `cumulative_rewards > break_score`
        self.if_keep_save = True  # keeping save the checkpoint. False means save until stop training.
        self.if_over_write = False  # overwrite the best policy network. `self.cwd/actor.pth`
        self.if_save_buffer = False  # if save the replay buffer for continuous training after stop training

        self.save_gap = int(8)  # save actor f"{cwd}/actor_*.pth" for learning curve.
        self.eval_times = int(3)  # number of times that get the average episodic cumulative return
        self.eval_per_step = int(2e4)  # evaluate the agent per training steps
        self.eval_env_class = None  # eval_env = eval_env_class(*eval_env_args)
        self.eval_env_args = None  # eval_env = eval_env_class(*eval_env_args)

    def init_before_training(self):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)

        '''set cwd (current working directory) for saving model'''
        if self.cwd is None:  # set cwd (current working directory) for saving model
            self.cwd = f'./{self.env_name}_{self.agent_class.__name__[5:]}_{self.random_seed}'

        '''remove history'''
        if self.if_remove is None:
            self.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
        if self.if_remove:
            import shutil
            shutil.rmtree(self.cwd, ignore_errors=True)
            print(f"| Arguments Remove cwd: {self.cwd}")
        else:
            print(f"| Arguments Keep cwd: {self.cwd}")
        os.makedirs(self.cwd, exist_ok=True)

    def get_if_off_policy(self) -> bool:
        agent_name = self.agent_class.__name__ if self.agent_class else ''
        on_policy_names = ('SARSA', 'VPG', 'A2C', 'A3C', 'TRPO', 'PPO', 'MPO')
        return all([agent_name.find(s) == -1 for s in on_policy_names])

    def print(self):
        from pprint import pprint
        pprint(vars(self))  # prints out args in a neat, readable format
