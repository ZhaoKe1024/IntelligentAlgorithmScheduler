#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/11/23 20:17
# @Author: ZhaoKe
# @File : BaseTrainer.py
# @Software: PyCharm
import os
import torch


class BaseTrainer(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        SEED = 3407
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        self.policy_net = None
        self.target_net = None
        self.optimizer = None

    def __save_checkpoint(self, save_model_path, epoch_id, best_model=False):
        policy_state_dict = self.policy_net.state_dict()
        target_state_dict = self.target_net.state_dict()
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
