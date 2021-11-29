import math
import numpy.random as rd
import sys
from copy import deepcopy
from net import ActorSAC, CriticTwin, Actor, Critic

from itertools import count
import gym
import numpy as np
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from tensorboardX import SummaryWriter
from drone_env import drone_env_block
from algorithms.ReplayMemory import ReplayMemory
import os
from OU import OU
from algorithms.vae import VAE
from algorithms.UnetMemory import UnetMemory
from algorithms.data_parallel import BalancedDataParallel
from torchvision import models
from PER import Memory
from algorithms.PrioritizedReplay import PrioritizedReplay
from algorithms.r2plus1d import R2Plus1DClassifier
from  algorithms.LRCN import LRCN
from algorithms.ConvLSTM import ConvLSTM
import cv2
from utils import soft_update
# from IPython.display import clear_output
# import matplotlib.pyplot as plt
# from matplotlib import animation
# from IPython.display import display
#
# %matplotlib inline
parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='test'
                                      '', type=str)  # mode = 'train' or 'test'
parser.add_argument("--env_name",
                    default="Drone1")  # OpenAI gym environment name， BipedalWalker-v2  Pendulum-v0
parser.add_argument('--tau', default=2.5e-3, type=float)  # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_episode', default=100, type=int)
parser.add_argument('--epoch', default=10, type=int)  # buffer采样的数据训练几次
parser.add_argument('--learning_rate', default=3e-8, type=float)
parser.add_argument('--gamma', default=0.99, type=int)  # discounted factor
parser.add_argument('--capacity', default=10000, type=int)# replay buffer size
parser.add_argument('--wait', default=129, type=int)
parser.add_argument('--num_episode', default=1000000, type=int)  # num of episodes in training
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--batch_size_unet', default=128, type=int)# mini batch size
parser.add_argument('--seed', default=True, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)

# optional parameters
# parser.add_argument('--num_hidden_layers', default=2, type=int)
# parser.add_\
# argument('--sample_frequency', default=256, type=int)
# parser.add_argument('--activation', default='Relu', type=str)
parser.add_argument('--render', default=True, type=bool)  # show UI or not
parser.add_argument('--log_interval', default=10, type=int)  # 每10episode保存一次模型
parser.add_argument('--load', default=False, type=bool)  # 训练前是否读取模型
parser.add_argument('--unet_load', default=False, type=bool)
parser.add_argument('--buffer_load', default=False, type=bool)
parser.add_argument('--render_interval', default=100, type=int)  # after render_interval, the env.render() will work
parser.add_argument('--policy_noise', default=0.2, type=float)  # 动作向量的噪声扰动的方差
parser.add_argument('--noise_clip', default=0.5, type=float)
parser.add_argument('--policy_delay', default=2, type=int)
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_frame', default=5000, type=int)
parser.add_argument('--print_log', default=1, type=int)
parser.add_argument('--seqsize', type=int, default=4)

parser.add_argument('--actor_lr',   type=float, default=5e-5)
parser.add_argument('--critic_lr',  type=float, default=2.5e-4)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
#device   = torch.device("cuda" if use_cuda else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)
directory = './exp' + script_name + args.env_name + './'
dir = './buffer' + script_name + args.env_name + './'
best_dir = './best' + script_name + args.env_name + './'
print(directory)

state_dim = 32#[None, 1, 72, 128]
action_dim = 2
max_action = 1  # 动作取值上界
train_indicator = 1
OU = OU()
gpu0_bsz = int(args.batch_size / 2 - 1)
acc_grad = 2
gpu0_unet = 15
acc_unet = 2
env = drone_env_block(aim=[58, 125, 10])


class AgentBase:
    def __init__(self):
        self.learning_rate = 1e-4
        self.soft_update_tau = 2 ** -8  # 5e-3 ~= 2 ** -8
        self.state = None  # set for self.update_buffer(), initialize before training
        self.device = None

        self.act = self.act_target = None
        self.cri = self.cri_target = None
        self.act_optimizer = None
        self.cri_optimizer = None
        self.criterion = None

    def init(self, net_dim, state_dim, action_dim):
        """initialize the self.object in `__init__()`

        replace by different DRL algorithms
        explict call self.init() for multiprocessing.

        :int net_dim: the dimension of networks (the width of neural networks)
        :int state_dim: the dimension of state (the number of state vector)
        :int action_dim: the dimension of action (the number of discrete action)
        """

    def select_action(self, state) -> np.ndarray:
        """Select actions for exploration

        :array state: state.shape==(state_dim, )
        :return array action: action.shape==(action_dim, ), (action.min(), action.max())==(-1, +1)
        """
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach_()
        action = self.act(states)[0]
        return action.cpu().numpy()

    def select_actions(self, states) -> np.ndarray:
        """Select actions for exploration

        :array states: (state, ) or (state, state, ...) or state.shape==(n, *state_dim)
        :return array action: action.shape==(-1, action_dim), (action.min(), action.max())==(-1, +1)
        """
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device).detach_()
        actions = self.act(states)
        return actions.cpu().numpy()  # -1 < action < +1

    def explore_env(self, env, buffer, target_step, reward_scale, gamma) -> int:
        """actor explores in env, then stores the env transition to ReplayBuffer

        :env: RL training environment. env.reset() env.step()
        :buffer: Experience Replay Buffer. buffer.append_buffer() buffer.extend_buffer()
        :int target_step: explored target_step number of step in env
        :float reward_scale: scale reward, 'reward * reward_scale'
        :float gamma: discount factor, 'mask = 0.0 if done else gamma'
        :return int target_step: collected target_step number of step in env
        """
        for _ in range(target_step):
            action = self.select_action(self.state)
            next_s, reward, done, _ = env.step(action)
            other = (reward * reward_scale, 0.0 if done else gamma, *action)
            buffer.append_buffer(self.state, other)
            self.state = env.reset() if done else next_s
        return target_step

    def update_net(self,  batch_size, repeat_times) -> (float, float):
        """update the neural network by sampling batch data from ReplayBuffer

        replace by different DRL algorithms.
        return the objective value as training information to help fine-tuning

        :buffer: Experience replay buffer. buffer.append_buffer() buffer.extend_buffer()
        :int target_step: explore target_step number of step in env
        :int batch_size: sample batch_size of data for Stochastic Gradient Descent
        :float repeat_times: the times of sample batch = int(target_step * repeat_times) in off-policy
        :return float obj_a: the objective value of actor
        :return float obj_c: the objective value of critic
        """

    def save_load_model(self, cwd, if_save):
        """save or load model files

        :str cwd: current working directory, we save model file here
        :bool if_save: save model or load model
        """
        act_save_path = '{}/actor.pth'.format(cwd)
        cri_save_path = '{}/critic.pth'.format(cwd)

        def load_torch_file(network, save_path):
            network_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
            network.load_state_dict(network_dict)

        if if_save:
            if self.act is not None:
                torch.save(self.act.state_dict(), act_save_path)
            if self.cri is not None:
                torch.save(self.cri.state_dict(), cri_save_path)
        elif (self.act is not None) and os.path.exists(act_save_path):
            load_torch_file(self.act, act_save_path)
            print("Loaded act:", cwd)
        elif (self.cri is not None) and os.path.exists(cri_save_path):
            load_torch_file(self.cri, cri_save_path)
            print("Loaded cri:", cwd)
        else:
            print("FileNotFound when load_model: {}".format(cwd))

    @staticmethod
    def soft_update(target_net, current_net, tau):
        """soft update a target network via current network

        :nn.Module target_net: target network update via a current network, it is more stable
        :nn.Module current_net: current network update via an optimizer
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data.__mul__(tau) + tar.data.__mul__(1 - tau))

class AgentSAC(AgentBase):
    def __init__(self):
        super().__init__()
        self.target_entropy = None
        self.alpha_log = None
        self.alpha_optimizer = None
        self.memory = ReplayMemory(args.capacity)
    def init(self, net_dim, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_entropy = np.log(action_dim)
        self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32,
                                      requires_grad=True, device=self.device)  # trainable parameter

        self.act = ActorSAC(net_dim, state_dim, action_dim).to(self.device)
        self.act = BalancedDataParallel(gpu0_bsz // acc_grad, self.act, dim=0)

        self.act_target = deepcopy(self.act)
        #self.act_target = BalancedDataParallel(gpu0_bsz // acc_grad, self.act_target, dim=0)

        self.cri = CriticTwin(int(net_dim * 1.25), state_dim, action_dim).to(self.device)
        self.cri = BalancedDataParallel(gpu0_bsz // acc_grad, self.cri, dim=0)

        self.cri_target = deepcopy(self.cri)
        #self.cri_target = BalancedDataParallel(gpu0_bsz // acc_grad, self.cri_target, dim=0)


        self.criterion = torch.nn.SmoothL1Loss()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)
        self.alpha_optimizer = torch.optim.Adam((self.alpha_log,), self.learning_rate)
        self.writer = SummaryWriter(directory)
        self.num_training = 0
        self.vae = VAE(device=device).to(device)
        self.vae = BalancedDataParallel(gpu0_bsz // acc_grad, self.vae, dim=0)
        self.um = UnetMemory(args.capacity)
        self.vae_update_iteration = 0
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)

    def select_action(self, state) -> np.ndarray:
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach_()
        action_hs = self.act.module.get_action(states)[0].cpu().data.numpy()
        action = np.zeros([3], dtype="float32")
        action[0] = 1.0
        action[1] = action_hs[0]
        action[2] = action_hs[1]
        # action[0] = action_hs[0]
        # action[1] = action_hs[1]
        # action[2] = action_hs[2]
        # action[3] = action_hs[3]
        return action

    def update_net(self, batch_size, repeat_times) -> (float, float):
        """Contribution of SAC (Soft Actor-Critic with maximum entropy)

        1. maximum entropy (Soft Q-learning -> Soft Actor-Critic, good idea)
        2. auto alpha (automating entropy adjustment on temperature parameter alpha for maximum entropy)
        3. SAC use TD3's TwinCritics too
        """
        #buffer.update_now_len_before_sample()

        alpha = self.alpha_log.exp().detach()
        obj_critic = None
        for _ in range(int(repeat_times)):
            '''objective of critic (loss function of critic)'''
            with torch.no_grad():
                #reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
                state, action, reward, next_state, mask = self.memory.sample(batch_size)
                state = torch.FloatTensor(np.float32(state)).to(device)
                next_s = torch.FloatTensor(np.float32(next_state)).to(device)
                action = torch.FloatTensor(action).to(device)
                reward = torch.FloatTensor(reward).to(device).unsqueeze(1)
                mask = torch.FloatTensor(mask).to(device).unsqueeze(1)

                next_a, next_logprob = self.act_target.module.get_action_logprob(next_s)
                next_q = torch.min(*self.cri_target.module.get_q1_q2(next_s, next_a))
                q_label = reward + mask * (next_q + next_logprob * alpha)
            q1, q2 = self.cri.module.get_q1_q2(state, action)
            obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
            self.cri_optimizer.zero_grad()
            obj_critic.backward()
            self.cri_optimizer.step()
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            '''objective of alpha (temperature parameter automatic adjustment)'''
            action_pg, logprob = self.act.module.get_action_logprob(state)  # policy gradient
            obj_alpha = (self.alpha_log * (logprob - self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            obj_alpha.backward()
            self.alpha_optimizer.step()

            '''objective of actor'''
            alpha = self.alpha_log.exp().detach()
            obj_actor = -(torch.min(*self.cri_target.module.get_q1_q2(state, action_pg)) + logprob * alpha).mean()

            self.act_optimizer.zero_grad()
            obj_actor.backward()
            self.act_optimizer.step()
            self.soft_update(self.act_target, self.act, self.soft_update_tau)
            self.writer.add_scalar('Loss/Qloss', obj_critic, global_step=self.num_training)
            self.writer.add_scalar('Loss/policy_loss', obj_actor, global_step=self.num_training)
            self.writer.add_scalar('Loss/Alpha_loss', obj_alpha, global_step=self.num_training)
            self.writer.add_scalar('entropy_temprature/alpha', alpha, global_step=self.num_training)
            self.num_training += 1
        return alpha.item(), obj_critic.item()
    def save(self, directory):
        torch.save(self.cri.state_dict(), directory + 'soft_q_net.pth')
        #torch.save(self.soft_q_net2.state_dict(), directory + 'soft_q_net2.pth')
        torch.save( self.cri_target.state_dict(), directory + 'target_soft_q_net.pth')
        #torch.save(self.target_soft_q_net2.state_dict(), directory + 'target_soft_q_net2.pth')
        torch.save(self.act.state_dict(), directory + 'policy_net.pth')
        #torch.save(self.critic_2_target.state_dict(), directory + 'critic_2_target.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self, directory):
        self.cri.load_state_dict(torch.load(directory + 'soft_q_net.pth'))
        #self.soft_q_net2.load_state_dict(torch.load(directory + 'soft_q_net2.pth'))
        self.cri_target.load_state_dict(torch.load(directory + 'target_soft_q_net.pth'))
        #self.target_soft_q_net2.load_state_dict(torch.load(directory + 'target_soft_q_net2.pth'))
        self.act.load_state_dict(torch.load(directory + 'policy_net.pth'))
        #self.critic_2_target.load_state_dict(torch.load(directory + 'critic_2_target.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")
    def VaeTrain(self, epochs):
        for epoch in range(epochs):
            images, depth = self.um.sample(args.batch_size_unet)
            images = torch.FloatTensor(images).to(device)
            depth = torch.FloatTensor(depth).to(device)
            recon_images, mu, logvar = self.vae(images)
            loss, bce, kld = self.vae.module.loss_fn(recon_images, depth, mu, logvar)
            self.vae_optimizer.zero_grad()
            self.writer.add_scalar('Loss/Vae_loss', loss, global_step=self.vae_update_iteration)
            self.vae_update_iteration += 1
            loss.backward()
            self.vae_optimizer.step()
    def save_vae(self):
        torch.save(self.vae.state_dict(), directory + 'vae.pth')
        print("====================================")
        print("Unet has been saved...")
        print("====================================")

    def load_vae(self):
        self.vae.load_state_dict(torch.load(directory + 'vae.pth'))
        print("====================================")
        print("Unet has been loaded...")
        print("====================================")
class OrnsteinUhlenbeckNoise:
    def __init__(self, size, theta=0.15, sigma=0.3, ou_noise=0.0, dt=1e-2):
        """The noise of Ornstein-Uhlenbeck Process
        Source: https://github.com/slowbull/DDPG/blob/master/src/explorationnoise.py
        It makes Zero-mean Gaussian Noise more stable.
        It helps agent explore better in a inertial system.
        Don't abuse OU Process. OU process has too much hyper-parameters and over fine-tuning make no sense.
        :int size: the size of noise, noise.shape==(-1, action_dim)
        :float theta: related to the not independent of OU-noise
        :float sigma: related to action noise std
        :float ou_noise: initialize OU-noise
        :float dt: derivative
        """
        self.theta = theta
        self.sigma = sigma
        self.ou_noise = ou_noise
        self.dt = dt
        self.size = size

    def __call__(self) -> float:
        """output a OU-noise
        :return array ou_noise: a noise generated by Ornstein-Uhlenbeck Process
        """
        noise = self.sigma * np.sqrt(self.dt) * rd.normal(size=self.size)
        self.ou_noise -= self.theta * self.ou_noise * self.dt + noise
        return self.ou_noise
class AgentDDPG(AgentBase):
    def __init__(self):
        super().__init__()
        self.ou_explore_noise = 0.3  # explore noise of action
        self.ou_noise = None

    def init(self, net_dim, state_dim, action_dim, if_per=False):
        self.ou_noise = OrnsteinUhlenbeckNoise(size=action_dim, sigma=self.ou_explore_noise)
        # I don't recommend to use OU-Noise
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = ReplayMemory(args.capacity)
        self.cri = Critic(net_dim, state_dim, action_dim).to(self.device)
        self.cri = BalancedDataParallel(gpu0_bsz // acc_grad, self.cri, dim=0)
        self.cri_target = deepcopy(self.cri)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)

        self.act = Actor(net_dim, state_dim, action_dim).to(self.device)
        self.act_target = deepcopy(self.act)
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)

        self.criterion = torch.nn.SmoothL1Loss(reduction='none' if if_per else 'mean')

        self.get_obj_critic = self.get_obj_critic_raw

    def select_action(self, state) -> np.ndarray:
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach_()
        action = self.act(states)[0].cpu().numpy()
        return (action + self.ou_noise()).clip(-1, 1)

    def update_net(self,  batch_size, repeat_times) -> (float, float):
        #buffer.update_now_len_before_sample()

        obj_critic = obj_actor = None  # just for print return
        for _ in range(int(repeat_times)):
            obj_critic, state = self.get_obj_critic(batch_size)
            self.cri_optimizer.zero_grad()
            obj_critic.backward()
            self.cri_optimizer.step()
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            q_value_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri_target(state, q_value_pg).mean()  # obj_actor
            self.act_optimizer.zero_grad()
            obj_actor.backward()
            self.act_optimizer.step()
            self.soft_update(self.act_target, self.act, self.soft_update_tau)
        return obj_actor.item(), obj_critic.item()

    def get_obj_critic_raw(self, batch_size):
        with torch.no_grad():
            # reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            state, action, reward, next_state, mask = self.memory.sample(batch_size)
            state = torch.FloatTensor(np.float32(state)).to(device)
            next_s = torch.FloatTensor(np.float32(next_state)).to(device)
            action = torch.FloatTensor(action).to(device)
            reward = torch.FloatTensor(reward).to(device).unsqueeze(1)
            mask = torch.FloatTensor(mask).to(device).unsqueeze(1)
            next_a = self.act_target(next_s)[0]  # policy noise
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics
            q_label = reward + mask * next_q
        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)  # twin critics
        return obj_critic, state

    # def get_obj_critic_per(self, buffer, batch_size):
    #     with torch.no_grad():
    #         reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
    #         next_q = self.cri_target(next_s, self.act_target(next_s))
    #         q_label = reward + mask * next_q
    #     q_value = self.cri(state, action)
    #     obj_critic = (self.criterion(q_value, q_label) * is_weights).mean()
    #
    #     td_error = (q_label - q_value.detach()).abs()
    #     buffer.td_error_update(td_error)
    #     return obj_critic, state


class AgentTD3(AgentDDPG):
    def __init__(self):
        super().__init__()
        self.explore_noise = 0.1  # standard deviation of explore noise
        self.policy_noise = 0.2  # standard deviation of policy noise
        self.update_freq = 2  # delay update frequency, for soft target update

    def init(self, net_dim, state_dim, action_dim, if_per=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = ReplayMemory(args.capacity)
        self.cri = CriticTwin(net_dim, state_dim, action_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)

        self.act = Actor(net_dim, state_dim, action_dim).to(self.device)
        self.act = BalancedDataParallel(gpu0_bsz // acc_grad, self.act, dim=0)
        self.act_target = deepcopy(self.act)
        self.vae = VAE(device=device).to(device)
        self.vae = BalancedDataParallel(gpu0_bsz // acc_grad, self.vae, dim=0)
        self.um = UnetMemory(args.capacity)
        self.vae_update_iteration = 0
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)

        self.criterion = torch.nn.SmoothL1Loss(reduction='none' if if_per else 'mean')
        #
        # if if_per:
        #     self.get_obj_critic = self.get_obj_critic_per
        # else:
        #     self.get_obj_critic = self.get_obj_critic_raw
        self.get_obj_critic = self.get_obj_critic_raw
        self.writer = SummaryWriter(directory)
        self.num_training = 0
    def select_action(self, state) -> np.ndarray:
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach_()
        action_hs = self.act(states)[0]
        action_hs = (action_hs + torch.randn_like(action_hs) * self.explore_noise).clamp(-1, 1)
        action = np.zeros([3], dtype="float32")
        action[0] = 1.0
        action[1] = action_hs[0]
        action[2] = action_hs[1]
        # action[0] = action_hs[0]
        # action[1] = action_hs[1]
        # action[2] = action_hs[2]
        # action[3] = action_hs[3]
        return action

    def update_net(self,  batch_size, repeat_times) -> (float, float):
        #buffer.update_now_len_before_sample()

        obj_critic = obj_actor = None
        for i in range(int(repeat_times)):
            obj_critic, state = self.get_obj_critic(batch_size)
            self.cri_optimizer.zero_grad()
            obj_critic.backward()
            self.cri_optimizer.step()
            if i % self.update_freq == 0:  # delay update
                self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            q_value_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri_target(state, q_value_pg).mean()  # obj_actor
            self.act_optimizer.zero_grad()
            obj_actor.backward()
            self.act_optimizer.step()
            if i % self.update_freq == 0:  # delay update
                self.soft_update(self.act_target, self.act, self.soft_update_tau)
            self.writer.add_scalar('Loss/Qloss', obj_critic, global_step=self.num_training)
            self.writer.add_scalar('Loss/policy_loss', obj_actor, global_step=self.num_training)
            self.num_training += 1
        return obj_actor.item(), obj_critic.item() / 2

    def get_obj_critic_raw(self,batch_size):
        with torch.no_grad():
            #reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            state, action, reward, next_state, mask = self.memory.sample(batch_size)
            state = torch.FloatTensor(np.float32(state)).to(device)
            next_s = torch.FloatTensor(np.float32(next_state)).to(device)
            action = torch.FloatTensor(action).to(device)
            reward = torch.FloatTensor(reward).to(device).unsqueeze(1)
            mask = torch.FloatTensor(mask).to(device).unsqueeze(1)
            next_a = self.act_target.module.get_action(next_s, self.policy_noise)  # policy noise
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics
            q_label = reward + mask * next_q
        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)  # twin critics
        return obj_critic, state

    # def get_obj_critic_per(self, buffer, batch_size):
    #     """Prioritized Experience Replay
    #     Contributor: Github GyChou
    #     """
    #     with torch.no_grad():
    #         #reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
    #         state, action, reward, next_state, mask, is_weights = self.memory.sample(batch_size)
    #         state = torch.FloatTensor(np.float32(state)).to(device)
    #         next_s = torch.FloatTensor(np.float32(next_state)).to(device)
    #         action = torch.FloatTensor(action).to(device)
    #         reward = torch.FloatTensor(reward).to(device).unsqueeze(1)
    #         mask = torch.FloatTensor(mask).to(device).unsqueeze(1)
    #         next_a = self.act_target.get_action(next_s, self.policy_noise)  # policy noise
    #         next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics
    #         q_label = reward + mask * next_q
    #
    #     q1, q2 = self.cri.get_q1_q2(state, action)
    #     obj_critic = ((self.criterion(q1, q_label) + self.criterion(q2, q_label)) * is_weights).mean()
    #
    #     td_error = (q_label - torch.min(q1, q2).detach()).abs()
    #     buffer.td_error_update(td_error)
    #     return obj_critic, state
    def save(self, directory):
        torch.save(self.cri.state_dict(), directory + 'soft_q_net.pth')
        #torch.save(self.soft_q_net2.state_dict(), directory + 'soft_q_net2.pth')
        torch.save( self.cri_target.state_dict(), directory + 'target_soft_q_net.pth')
        #torch.save(self.target_soft_q_net2.state_dict(), directory + 'target_soft_q_net2.pth')
        torch.save(self.act.state_dict(), directory + 'policy_net.pth')
        #torch.save(self.critic_2_target.state_dict(), directory + 'critic_2_target.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self, directory):
        self.cri.load_state_dict(torch.load(directory + 'soft_q_net.pth'))
        #self.soft_q_net2.load_state_dict(torch.load(directory + 'soft_q_net2.pth'))
        self.cri_target.load_state_dict(torch.load(directory + 'target_soft_q_net.pth'))
        #self.target_soft_q_net2.load_state_dict(torch.load(directory + 'target_soft_q_net2.pth'))
        self.act.load_state_dict(torch.load(directory + 'policy_net.pth'))
        #self.critic_2_target.load_state_dict(torch.load(directory + 'critic_2_target.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")
    def VaeTrain(self, epochs):
        for epoch in range(epochs):
            images, depth = self.um.sample(args.batch_size_unet)
            images = torch.FloatTensor(images).to(device)
            depth = torch.FloatTensor(depth).to(device)
            recon_images, mu, logvar = self.vae(images)
            loss, bce, kld = self.vae.module.loss_fn(recon_images, depth, mu, logvar)
            self.vae_optimizer.zero_grad()
            self.writer.add_scalar('Loss/Vae_loss', loss, global_step=self.vae_update_iteration)
            self.vae_update_iteration += 1
            loss.backward()
            self.vae_optimizer.step()
    def save_vae(self):
        torch.save(self.vae.state_dict(), directory + 'vae.pth')
        print("====================================")
        print("Unet has been saved...")
        print("====================================")

    def load_vae(self):
        self.vae.load_state_dict(torch.load(directory + 'vae.pth'))
        print("====================================")
        print("Unet has been loaded...")
        print("====================================")
def main():
    agent = AgentSAC()
    #agent = AgentTD3()
    agent.init(2 ** 8, state_dim, action_dim)
    lv_0 = 0
    lv_1 = 0
    lv_2 = 0
    lv_3 = 0
    lv_4 = 0
    lv_5 = 0
    lv_6 = 0
    lv_7 = 0
    lv_8 = 0
    lv_9 = 0
    lv_10 = 0
    lv_11 = 0
    ep_r = 0
    pointer = 0
    if args.mode == 'test':
        agent.load(directory)
        agent.load_vae()
        epsilon_model = 'test'
        for epoch in range(args.test_episode):  # 50
            s = env.reset(env_switch=False)
            # state = torch.FloatTensor(s).to(device)
            # state = agent.unet(state).cpu().data.numpy()
            s = torch.FloatTensor(s).to(device)
            state, _, _ = agent.vae.module.encode(s)
            state = state.cpu().data.numpy().flatten()
            state = np.stack([state] * args.seqsize, axis=0)
            for t in count():
                action = agent.select_action(state)
                print("Action: {} ".format(action))
                s_, reward, done, info, labdepth, lv_end = env.step(action)
                # next_state = torch.FloatTensor(s_).to(device)
                # next_state = agent.unet(next_state).cpu().data.numpy()
                s_ = torch.FloatTensor(s_).to(device)
                next_state, _, _ = agent.vae.module.encode(s_)
                next_state = next_state.cpu().data.numpy().flatten()
                next_state = np.append(state[1:], [next_state], axis=0)
                vaeshow, _, _ = agent.vae(s_)
                vaeshow = vaeshow.detach().cpu().numpy()
                vaeshow = np.array(vaeshow * 255, dtype=np.uint8)
                cv2.imshow('Unet_Out', vaeshow[0][0])
                k = cv2.waitKey(1) & 0xff
                ep_r += reward
                #env.render()
                if done or t == args.max_frame - 1:
                    if lv_end == 0:
                        lv_0 = lv_0 + 1
                        print("lv: %r" %(lv_end))
                    elif lv_end == 1:
                        lv_1 = lv_1 + 1
                        print("lv: %r" %(lv_end))
                    elif lv_end == 2:
                        lv_2 = lv_2 + 1
                        print("lv: %r" %(lv_end))
                    elif lv_end == 3:
                        lv_3 = lv_3 + 1
                        print("lv: %r" %(lv_end))
                    elif lv_end == 4:
                        lv_4 = lv_4 + 1
                        print("lv: %r" %(lv_end))
                    elif lv_end == 5:
                        lv_5 = lv_5 + 1
                        print("lv: %r" % (lv_end))
                    elif lv_end == 6:
                        lv_6 = lv_6 + 1
                        print("lv: %r" % (lv_end))
                    elif lv_end == 7:
                        lv_7 = lv_7 + 1
                        print("lv: %r" % (lv_end))
                    elif lv_end == 8:
                        lv_8 = lv_8 + 1
                        print("lv: %r" % (lv_end))
                    elif lv_end == 9:
                        lv_9 = lv_9 + 1
                        print("lv: %r" % (lv_end))
                    elif lv_end == 10:
                        lv_10 = lv_10 + 1
                        print("lv: %r" % (lv_end))
                    else:
                        lv_11 = lv_11 + 1
                        print("lv: %r" % (lv_end))
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(epoch, ep_r, t))
                    ep_r = 0
                    break
                state = next_state
        print("lv1: {} lv2: {} lv3: {} lv4: {} lv5: {} lv6: {} lv7: {} lv8: {} lv9: {} lv10: {} lv11: {}".format(lv_1, lv_2, lv_3,lv_4,lv_5,lv_6,lv_7,lv_8,lv_9,lv_10,lv_11))

    elif args.mode == 'train':
        print("====================================")
        print("Collection Experience...")
        print("====================================")
        if args.load: agent.load(directory)
        if args.unet_load: agent.load_vae()
        if args.buffer_load: agent.memory = agent.memory.load(dir)
        epsilon_model = 'test'
        count_epoch = 0
        sum_r = []
        sum_s = []
        mean_r = -3
        for epoch in range(args.num_episode):
            if pointer <= args.capacity - 1 :
                agent.memory.save(dir)
            if len(agent.memory.sucBuffer) + len(agent.memory.buffer) >= args.wait - 1:
                replay_max = args.capacity
                replay_len = len(agent.memory.sucBuffer) + len(agent.memory.buffer)

                # k = 1 + replay_len / replay_max
                #
                # batch_size = int(k * args.batch_size)
                # update_times = int(k * args.epoch)
                batch_size = args.batch_size
                update_times = args.epoch
                agent.update_net(batch_size, update_times)

                print("learn")
            # if len(agent.um.buffer) >= args.batch_size_unet and  len(agent.um.buffer)<= 10000:
            if len(agent.um.buffer) >= args.batch_size_unet:
                agent.VaeTrain(10)
                print("vae learn")

            count_epoch = epoch
            if count_epoch % 10 == 0 and count_epoch != 0:
                env_switch = True
            else:
                env_switch = False

            s = env.reset(env_switch)





            s = torch.FloatTensor(s).to(device)
            state, _, _ = agent.vae.module.encode(s)
            state = state.cpu().data.numpy().flatten()
            vaes, _, _ = agent.vae(s)
            vaes = vaes.detach().cpu().numpy()
            in_map = s.detach().cpu().numpy()

            # state = agent.unet(state).cpu().data.numpy()
            state = np.stack([state] * args.seqsize, axis=0)

            if pointer % 100 == 0:
                dir_in = 'input' + str(epoch) + '.png'
                dir_out = 'output' + str(epoch) + '.png'
                in_map = np.array(in_map * 255, dtype=np.uint8)[0]
                in_map = np.reshape(in_map, (72, 128, 3))
                out_map = np.array(vaes * 255, dtype=np.uint8)[0][0]
                cv2.imwrite("C:\\Vae_img\\"+dir_in, in_map)
                cv2.imwrite("C:\\Vae_img\\" + dir_out, out_map)

            ep_r = 0
            ep_s = 0
            for t in range(args.max_frame):

                action = agent.select_action(state)
                #action = action + np.random.normal(0, args.exploration_noise, size=action_dim)
                #action = action.clip(-max_action, max_action)
                print("Action: {}  pointer: {} step: {}".format(action, pointer, t))
                s_, reward, done, info, labdepth, level_end = env.step(action)
                if info == "success":
                    ep_s +=1
                # if info == "through":
                #     s_[:] = 0
                #     print("camera go through the wall!!!")
                #action = np.array([1, -0.2, 0]).astype(np.float32)
                # reward_3 = 0
                # action_3 = action.reshape(3, 3)
                # suc = False
                # for i in range(3):
                #     action_i = action_3[i]
                #     s_, reward, done, info, labdepth, level_end  = env.step(action_i)
                #     next_state = s_
                #     reward_3 = reward_3 + reward
                #     done_3 = done
                #     if done_3:
                #         break
                #     if info == "success":
                #         suc = True
                # if suc:
                #     info = "success"
                # print("R: %r" %(reward_3))


                in_s = env.getImg()[0]
                out_s = env.getDepth()[0]
                agent.um.append(in_s, out_s)
                s_ = torch.FloatTensor(s_).to(device)

                # next_state = agent.unet(next_state).cpu().data.numpy()
                next_state, _, _ = agent.vae.module.encode(s_)
                next_state = next_state.cpu().data.numpy().flatten()
                next_state = np.append(state[1:], [next_state], axis=0)
                vaeshow,_,_ = agent.vae(s_)
                vaeshow = vaeshow.detach().cpu().numpy()
                vaeshow = np.array(vaeshow * 255, dtype=np.uint8)
                cv2.imshow('Unet_Out', vaeshow[0][0])
                k = cv2.waitKey(1) & 0xff
                ep_r += reward
                # if args.render and epoch >= args.render_interval:
                #     env.render()
                #agent.memory.push((state, next_state, action, reward, np.float(done)))
                index = [0]

                new_a = np.delete(action, index)
                #agent.memory.push(state, new_a, reward, next_state, np.float(done))
                mask = 0.0 if done else args.gamma
                agent.memory.append(state, new_a, reward, next_state, mask, info)
                pointer += 1
                state = next_state

                #learn
                # if len(agent.memory.sucBuffer) + len(agent.memory.buffer) >= args.wait - 1:
                #     replay_max = args.capacity
                #     replay_len = len(agent.memory.sucBuffer) + len(agent.memory.buffer)
                #
                #     # k = 1 + replay_len / replay_max
                #     #
                #     # batch_size = int(k * args.batch_size)
                #     # update_times = int(k * args.epoch)
                #     batch_size = args.batch_size
                #     update_times = args.epoch
                #     agent.update_net(update_times, batch_size)
                #
                #     print("learn")
                #     # if len(agent.um.buffer) >= args.batch_size_unet and  len(agent.um.buffer)<= 10000:
                # if len(agent.um.buffer) >= args.batch_size_unet and epoch % 20 == 0:
                #     agent.VaeTrain(10)
                #     print("vae learn")


                if done or t == args.max_frame - 1:
                    agent.writer.add_scalar('ep_r', ep_r, global_step=epoch)
                    if len(sum_s) == 50:
                        agent.writer.add_scalar('Average number of obstacle avoidance in 50 epochs', np.mean(sum_s), global_step=epoch)
                    if epoch % args.print_log == 0:
                        print("Ep_i {}, the ep_r is {:0.2f}, the step is {}".format(epoch, ep_r, t))

                    break

            sum_r.append(ep_r)
            if len(sum_r) > 10:
                sum_r.pop(0)
            if len(sum_r) == 10:
                if np.mean(sum_r) >= mean_r:
                    agent.save(best_dir)
                    mean_r = np.mean(sum_r)
            if epoch % args.log_interval == 0:
                    agent.save(directory)
                    agent.save_vae()
                #agent.save_unet()

            sum_s.append(ep_s)

            if len(sum_s) > 50:
                sum_s.pop(0)



    else:
        raise NameError("mode wrong!!!")

if __name__ == '__main__':
    # replay_buffer = ReplayBuffer(int(1e6))
    #
    # #env = NormalizedActions(gym.make("Pendulum-v0"))
    # env = drone_env_block(aim=[58, 125, 10])
    #
    # agent = SAC(env, replay_buffer)
    #
    # train(agent)
    main()