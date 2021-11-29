import math
import random
import sys
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
from algorithms.U_net import Unet
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

parser.add_argument('--mode', default='train', type=str)  # mode = 'train' or 'test'
parser.add_argument("--env_name",
                    default="Drone1")  # OpenAI gym environment name， BipedalWalker-v2  Pendulum-v0
parser.add_argument('--tau', default=2.5e-3, type=float)  # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_episode', default=100, type=int)
parser.add_argument('--epoch', default=10, type=int)  # buffer采样的数据训练几次
parser.add_argument('--learning_rate', default=3e-8, type=float)
parser.add_argument('--gamma', default=0.93, type=int)  # discounted factor
parser.add_argument('--capacity', default=100000, type=int)# replay buffer size
parser.add_argument('--wait', default=500, type=int)
parser.add_argument('--num_episode', default=1000000, type=int)  # num of episodes in training
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--batch_size_unet', default=32, type=int)# mini batch size
parser.add_argument('--seed', default=True, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)

# optional parameters
# parser.add_argument('--num_hidden_layers', default=2, type=int)
# parser.add_\
# argument('--sample_frequency', default=256, type=int)
# parser.add_argument('--activation', default='Relu', type=str)
parser.add_argument('--render', default=True, type=bool)  # show UI or not
parser.add_argument('--log_interval', default=50, type=int)  # 每10episode保存一次模型
parser.add_argument('--load', default=False, type=bool)  # 训练前是否读取模型
parser.add_argument('--unet_load', default=False, type=bool)
parser.add_argument('--buffer_load', default=True, type=bool)
parser.add_argument('--render_interval', default=100, type=int)  # after render_interval, the env.render() will work
parser.add_argument('--policy_noise', default=0.2, type=float)  # 动作向量的噪声扰动的方差
parser.add_argument('--noise_clip', default=0.5, type=float)
parser.add_argument('--policy_delay', default=2, type=int)
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_frame', default=5000, type=int)
parser.add_argument('--print_log', default=1, type=int)
parser.add_argument('--seqsize', type=int, default=5)

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

state_dim = 32#[None, 1, 8, 144, 256]
action_dim = 2
max_action = 1  # 动作取值上界
train_indicator = 1
OU = OU()
gpu0_bsz = int(args.batch_size / 2 - 1)
acc_grad = 2
gpu0_unet = 15
acc_unet = 2
env = drone_env_block(aim=[58, 125, 10])





class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=[64, 25], init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(9216 + num_actions, 4096)
        self.fc0 = nn.Linear(4096, 2048)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)
        # self.linear2 = nn.Linear(num_inputs + num_actions, 2048)
        # self.fc5 = nn.Linear(2048, 1024)
        # self.fc6 = nn.Linear(1024, 256)
        # self.fc7 = nn.Linear(256, hidden_size[1])
        # self.fc8 = nn.Linear(hidden_size[1], 1)

        self.fc5.weight.data.uniform_(-init_w, init_w)
        self.fc5.bias.data.uniform_(-init_w, init_w)
        #
        # self.fc8.weight.data.uniform_(-init_w, init_w)
        # self.fc8.bias.data.uniform_(-init_w, init_w)


    def forward(self, state, action):
        f = state.view(-1, self.num_flat_features(state))
        x = torch.cat([f, action], 1)
        # x1 = F.relu(self.linear1(x))
        # x1 = F.relu(self.fc0(x1))
        # x1 = F.relu(self.fc1(x1))
        # x1 = F.relu(self.fc2(x1))
        x = self.linear1(x)
        x1 = F.relu(self.fc0(x))
        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))
        x1 = F.relu(self.fc4(x1))
        x1 = F.relu(self.fc5(x1))
        # x2 = F.relu(self.linear2(x))
        # x2 = F.relu(self.fc5(x2))
        # x2 = F.relu(self.fc6(x2))
        # x2 = F.relu(self.fc7(x2))
        # x2 = F.relu(self.fc8(x2))

        return x1

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1

        for s in size:
            num_features *= s

        return num_features

class VideoNet(nn.Module):
    def __init__(self):
        super(VideoNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(256, 64)
    def forward(self, state):


        x = F.max_pool3d(F.relu(self.conv1(state)), (2, 3, 3))
        x = F.max_pool3d(F.relu(self.conv2(x)), (2, 3, 3))
        f = self.pool(x)
        f = f .view(-1, self.num_flat_features(f))
        return f

class ImageNet(nn.Module):
    def __init__(self,num_classes=1000):
        super(ImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    #     self.alex = models.AlexNet()
    #     # self.conv1 = nn.Conv2d(1, 16, kernel_size=(3,3), stride=1, padding=1)
    #     # self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1)
    #     # self.conv3 = nn.Conv2d(32, 16, kernel_size=(3, 3), stride=1, padding=1)
    #     # self.conv4 = nn.Conv2d(16, 8, kernel_size=(3, 3), stride=1, padding=1)
    #     # self.pool = nn.AdaptiveAvgPool3d(1)
    #     # self.fc = nn.Linear(1000, 64)
    #
    # def forward(self, state):
    #     x = self.alex(state)
    #     # x = F.max_pool2d(F.relu(self.conv1(state)), (2, 2))
    #     # x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
    #     # x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
    #     # x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
    #     # f = x.view(-1, self.num_flat_features(x))
    #     return x
    # def num_flat_features(self, x):
    #     size = x.size()[1:]  # all dimensions except the batch dimension
    #     num_features = 1
    #
    #     for s in size:
    #         num_features *= s
    #
    #     return num_features
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=[64, 25],
                 init_w=3e-3, log_std_min=-20, log_std_max=2, epsilon=1e-6):
        super(PolicyNetwork, self).__init__()

        self.epsilon = epsilon

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        # self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])

        # self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding=1)
        # self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1)
        # # self.gru = nn.GRU(2880, 200)
        # self.norm1 = nn.BatchNorm2d(1)
        # self.norm2 = nn.BatchNorm2d(64)
        self.x_dense = nn
        # self.R21Dmodel = R2Plus1DClassifier(hidden_size[0], (2, 2, 2, 2))
        # self.R21Dmodel = BalancedDataParallel(gpu0_bsz // acc_grad, self.R21Dmodel, dim=0)
        # self.R21Dmodel = VideoNet()
        # self.R21Dmodel = BalancedDataParallel(gpu0_bsz // acc_grad, self.R21Dmodel, dim=0).cuda()
        # self.R21Dmodel =    LRCN(input_dim=512, hidden_dim=512, num_layers=2, bidirectional=True, num_classes
        #     =hidden_size[0], device = device)
        #self.convLSTM = ConvLSTM(input_dim=1, hidden_dim=64, kernel_size=(3,3), num_layers=2)
        #self.fc0 = nn.Linear(hidden_size[0],  hidden_size[1])
        # self.fc1 = nn.Linear(2048,  1024)
        # self.fc2 = nn.Linear(1024, 256)
        # self.fc3 = nn.Linear(32, hidden_size[1])
        #self.fc3 = nn.Linear(200, action_dim)
        self.fc  = nn.Linear(9216, 4096)
        #self.imgnet = ImageNet().to(device)
        self.fc0 = nn.Linear(4096, 2048)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.pool = nn.AdaptiveAvgPool2d(64)
        self.max_action = max_action
        self.min_action = - max_action

        self.x_mean_linear = nn.Linear(64, 1)
        self.mean_linear = nn.Linear(64, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(64, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_scale = torch.tensor(1.)
        self.action_bias = torch.tensor(0.)
    def forward(self, state, deterministic=False):
        # x = F.relu(self.linear1(state))
        # x = F.relu(self.linear2(x))

        #x = self.norm1(state)
        # x = F.max_pool2d(F.relu(self.conv1(state)), (3, 3))
        #
        # x = F.max_pool2d(F.relu(self.conv2(x)), (3, 3))
        # x = F.max_pool2d(F.relu(self.conv3(x)), (3, 3))
        #x = self.norm2(x)
        #f = self.imgnet(state)
        f = state.view(-1, self.num_flat_features(state))
        #f = self.R21Dmodel(state)
        a = F.relu(self.fc(f))
        a = F.relu(self.fc0(a))
        a = F.relu(self.fc1(a))
        a = F.relu(self.fc2(a))
        a = F.relu(self.fc3(a))
        a = F.relu(self.fc4(a))
        ##a = self.pool(f)
        # x_mean = self.x_mean_linear(a)
        # yz_mean = self.mean_linear(a)
        mean = self.mean_linear(a)
        #mean = torch.cat(inputs=( x_mean, yz_mean), dimension=1)

        log_std = self.log_std_linear(a)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        std = torch.exp(log_std)

        #log_prob = None

        # if deterministic:
        #     action = torch.tanh(mean)
        # else:
        #     # assumes actions have been normalized to (0,1)
        #     normal = Normal(0, 1)
        #     z = mean + std * normal.sample().requires_grad_()
        #     action = torch.tanh(z)
        #
        #     log_prob = Normal(mean, std).log_prob(z) - torch.log(1 - action * action + self.epsilon)
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, mean, log_std, log_prob, std

    def get_action(self, state, deterministic=False):
        #state = torch.FloatTensor(state).unsqueeze(0).to(device)
        #state = torch.tensor(state).float().to(device)
        state = state.clone().detach().float().to(device)
        action, _, _, _, _ = self.forward(state, deterministic)
        act = action
        return act

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1

        for s in size:
            num_features *= s

        return num_features

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action


def normalize_action(action, low, high):
    action = low + (action + 1.0) * 0.5 * (high - low)
    action = np.clip(action, low, high)

    return action


class SAC(object):

    def __init__(self, env,  state_dim, action_dim, seed=0, hidden_dim=[256, 128],
                 steps_per_epoch=200, epochs=1000, discount=0.95,
                 tau=0.0005, lr=3e-5, auto_alpha=True, batch_size=100, start_steps=10000,
                 max_ep_len=200, logger_kwargs=dict(), save_freq=1):

        # Set seeds
        self.env = env
        #self.env.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # env space
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # init networks

        # Soft Q


        self.soft_q_net1 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.soft_q_net1 = BalancedDataParallel(gpu0_bsz // acc_grad, self.soft_q_net1, dim=0)
        self.soft_q_net2 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.soft_q_net2 = BalancedDataParallel(gpu0_bsz // acc_grad, self.soft_q_net2, dim=0)
        self.target_soft_q_net1 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.target_soft_q_net1 = BalancedDataParallel(gpu0_bsz // acc_grad, self.target_soft_q_net1, dim=0)
        self.target_soft_q_net2 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.target_soft_q_net2 = BalancedDataParallel(gpu0_bsz // acc_grad, self.target_soft_q_net2, dim=0)
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)

        # for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
        #     target_param.data.copy_(param.data)

        # Policy
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(device)

        # Optimizers/Loss
        self.soft_q_criterion = nn.MSELoss()

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # alpha tuning
        self.auto_alpha = auto_alpha

        if self.auto_alpha:
            self.target_entropy = -np.prod(action_dim).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        #self.replay_buffer = replay_buffer
        self.discount = discount
        self.batch_size = batch_size
        self.tau = tau
        self.memory = ReplayMemory(args.capacity)
        #self.memory = Memory(args.capacity)
        #self.memory = PrioritizedReplay(capacity=args.capacity)
        self.um = UnetMemory(args.capacity)
        self.writer = SummaryWriter(directory)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0
        self.unet = Unet(1, 1).to(device)
        self.unet_update_iteration = 0
        self.unet =  BalancedDataParallel(gpu0_unet // acc_unet, self.unet, dim=0)
        self.alpha = 0.2
    def get_action(self, state, deterministic=False, explore=False):

        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if explore:
            return self.env.action_space.sample()
        else:
            action = self.policy_net.get_action(state, deterministic).detach()
            return action.numpy()

    def  select_action(self, state, wait_before_train, step, epsilon_model):
        epsilon = 1
        noise = True
        state = torch.tensor(state).float().to(device)
        action_hs = self.policy_net.get_action(state).cpu().data.numpy().flatten()
        action = np.zeros([3], dtype="float32")
        action[0] = 0.6
        action[1] = action_hs[0]
        action[2] = action_hs[1]
        # action[3] = 0.5
        # action[4] = action_hs[2]
        # action[5] = action_hs[3]
        # action[6] = 0.5
        # action[7] = action_hs[4]
        # action[8] = action_hs[5]
        action_truth = np.zeros([3])
        noise_t = np.zeros([action_dim])
        epsilon_ceil = 0.95
        if epsilon_model == 'test' or epsilon_model == 'train':
            noise = False
        elif epsilon_model == 'linear':
            epsilon = epsilon_ceil * (step - wait_before_train) / (20000 - wait_before_train)
            if epsilon > epsilon_ceil:
                epsilon = epsilon_ceil
        elif epsilon_model == 'exponential':
            epsilon = 1 - math.exp(-2 / (100000 - wait_before_train) * (step - wait_before_train))
            if epsilon > epsilon_ceil:
                epsilon = epsilon_ceil
        if epsilon_model != 'test':
            if random.random() > epsilon:
                noise = True
            else:
                noise = False
        if noise:
            action_truth[0] = action[0]
            action_truth[1] = action[1]
            action_truth[2] = action[2]

            #noise_t[0] = train_indicator * OU.function(action[0], 0.5, 0.6, 0.10)
            noise_t[0] = train_indicator * OU.function(action[1], 0.0, 0.6, 0.30)
            noise_t[1] = train_indicator * OU.function(action[2], 0.0, 0.6, 0.30)

            #action[0] = np.clip(action[0]+noise_t[0], -max_action, max_action)
            action[1] = np.clip(action[1] + noise_t[0], -max_action, max_action)
            action[2] = np.clip(action[2] + noise_t[1], -max_action, max_action)


        else:
            # for i in range(3):
            #     action[i] = np.clip(action[i], -self.a_bound, self.a_bound)
            action_truth[0] = action[0]
            action_truth[1] = action[1]
            action_truth[2] = action[2]
            # action_truth[3] = action[3]
            # action_truth[4] = action[4]
            # action_truth[5] = action[5]
            # action_truth[6] = action[6]
            # action_truth[7] = action[7]
            # action_truth[8] = action[8]
            # noise_t[0] =train_indicator * OU.function(action[0], 0.5, 0.06, 0.10)
            # noise_t[0] = train_indicator * OU.function(action[1], 0.0, 0.06, 0.30)
            # noise_t[1] = train_indicator * OU.function(action[2], 0.0, 0.06, 0.30)

            #action[0] = np.clip(action[0]+noise_t[0], -max_action, max_action)
            # action[1] = np.clip(action[1] + noise_t[0], -max_action, max_action)
            # action[2] = np.clip(action[2] + noise_t[1], -max_action, max_action)
        return action, noise_t, action_truth

    def update(self, iterations, batch_size):

        for _ in range(0, iterations):

            #experiences = self.memory.sample(batch_size)
            state, action, reward, next_state, done = self.memory.sample(batch_size)

            #states, actions, rewards, next_states, dones, idx, weights = experiences
            # state = torch.FloatTensor(state).to(device)
            # next_state = torch.FloatTensor(next_state).to(device)
            # action = torch.FloatTensor(action).to(device)
            # #action_x, action = action.split([1, 2], dim=1)
            # reward = torch.FloatTensor(reward).to(device)
            # done = torch.FloatTensor(np.float32(done)).to(device)

            state = torch.FloatTensor(np.float32(state)).to(device)
            next_state = torch.FloatTensor(np.float32(next_state)).to(device)
            action = torch.FloatTensor(action).to(device)
            reward = torch.FloatTensor(reward).to(device).unsqueeze(1)
            done = torch.FloatTensor(done).to(device).unsqueeze(1)
            #weights = torch.FloatTensor(weights).unsqueeze(1)

            # with torch.no_grad():
            #     next_state_action, _, _,next_state_log_pi, *_ = self.policy_net(next_state)
            #     qf1_next_target, qf2_next_target = self.target_soft_q_net(next_state, next_state_action)
            #     min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            #     next_q_value = reward + done * self.discount * (min_qf_next_target)
            # qf1, qf2 = self.soft_q_net(state,
            #                        action)  # Two Q-functions to mitigate positive bias in the policy improvement step
            # qf1_loss = F.mse_loss(qf1,
            #                       next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            # qf2_loss = F.mse_loss(qf2,
            #                       next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            # qf_loss = qf1_loss + qf2_loss
            #
            # self.soft_q_optimizer1.zero_grad()
            # qf_loss.backward()
            # self.soft_q_optimizer1.step()
            #
            # _, pi, log_pi, _, std = self.policy_net(state)
            #
            # qf1_pi, qf2_pi = self.soft_q_net(state, pi)
            # min_qf_pi = torch.min(qf1_pi, qf2_pi)
            #
            # policy_loss = ((
            #                            self.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
            #
            # self.policy_optimizer.zero_grad()
            # policy_loss.backward()
            # self.policy_optimizer.step()
            #
            # if self.auto_alpha:
            #     alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            #
            #     self.alpha_optimizer.zero_grad()
            #     alpha_loss.backward()
            #     self.alpha_optimizer.step()
            #
            #     self.alpha = self.log_alpha.exp()
            #     alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
            # else:
            #     alpha_loss = torch.tensor(0.).to(self.device)
            #     alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs
            #
            # if self.num_training % 1 == 0:
            #     soft_update(self.target_soft_q_net, self.soft_q_net, self.tau)
            # # kais
            new_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy_net(state)

            if self.auto_alpha:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                alpha = self.log_alpha.exp()

            else:
                alpha_loss = 0
                alpha = 0.2  # constant used by OpenAI

            # Update Policy
            #state_f = new_actions[0]
            # state_t = self.policy_net.R21Dmodel(state)
            #state_t = self.policy_net.convLSTM(state)
            # state_t = self.policy_net.imgnet(state)

            q_new_actions = torch.min(
                self.soft_q_net1(state, new_actions),
                self.soft_q_net2(state, new_actions)
            )

            #policy_loss = (alpha * log_pi - q_new_actions).mean()

            # Update Soft Q Function

            q1_pred = self.soft_q_net1(state, action)
            q2_pred = self.soft_q_net2(state, action)

            new_next_actions, _, _, new_log_pi, *_ = self.policy_net(next_state)
            #next_state_f = new_next_actions[0]
            #next_state_t = self.policy_net.R21Dmodel(next_state)
            #next_state_t = self.policy_net.convLSTM(next_state)
            #next_state_t = self.policy_net.imgnet(next_state)
            target_q_values = torch.min(

                self.target_soft_q_net1(next_state, new_next_actions),
                self.target_soft_q_net2(next_state, new_next_actions),
            ) - alpha * new_log_pi

            q_target = reward + (1 - done) * self.discount * target_q_values
            td_error1 = q_target.detach() - q1_pred  # ,reduction="none"
            td_error2 = q_target.detach() - q2_pred
            q1_loss = self.soft_q_criterion(q1_pred, q_target.detach())

            q2_loss = self.soft_q_criterion(q2_pred, q_target.detach())
            prios = abs(((td_error1 + td_error2) / 2.0 + 1e-5).squeeze())


            # Update Networks
            self.soft_q_optimizer1.zero_grad()
            q1_loss.backward(retain_graph=True)


            self.soft_q_optimizer2.zero_grad()
            q2_loss.backward(retain_graph=True)


            policy_loss = (alpha * log_pi - q_new_actions).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.soft_q_optimizer1.step()
            self.soft_q_optimizer2.step()
            self.policy_optimizer.step()

            #self.memory.update_priorities(idx, prios.data.cpu().numpy())
            # Soft Updatesvc
            for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.tau) + param.data * self.tau
                )

            for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.tau) + param.data * self.tau
                )
            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

            #critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
            self.writer.add_scalar('Loss/Q1_loss', q1_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/Q2_loss', q2_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/policy_loss', policy_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/Alpha_loss', alpha_loss, global_step=self.num_training)
            self.writer.add_scalar('entropy_temprature/alpha', alpha, global_step=self.num_training)
            self.num_training += 1

    def append_memory(self, state, action, reward, next_state, done, info):
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(np.array([action])).to(device)
        # action_x, action = action.split([1, 2], dim=1)
        reward = torch.FloatTensor(np.array([np.float32(reward)])).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.array([np.float32(done)])).unsqueeze(1).to(device)
        #info = torch.FloatTensor(np.array([np.float32(info)])).unsqueeze(1).to(device)
        Q = torch.min(
                self.soft_q_net1(state, action),
                self.soft_q_net2(state, action)
            )

        target_action, _, _, new_log_pi, *_ = self.policy_net(next_state)
        alpha = self.log_alpha.exp()
        target_Q = torch.min(

                self.target_soft_q_net1(next_state, target_action),
                self.target_soft_q_net2(next_state, target_action),
            ) - alpha * new_log_pi
        td = reward + (1 - done) * self.discount * target_Q - Q
        td = float(abs(td[0]))
        self.memory.add(td, (state, action, reward, next_state, done))
        return td

    def save(self, directory):
        torch.save(self.soft_q_net1.state_dict(), directory + 'soft_q_net1.pth')
        torch.save(self.soft_q_net2.state_dict(), directory + 'soft_q_net2.pth')
        torch.save(self.target_soft_q_net1.state_dict(), directory + 'target_soft_q_net1.pth')
        torch.save(self.target_soft_q_net2.state_dict(), directory + 'target_soft_q_net2.pth')
        torch.save(self.policy_net.state_dict(), directory + 'policy_net.pth')
        #torch.save(self.critic_2_target.state_dict(), directory + 'critic_2_target.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self, directory):
        self.soft_q_net1.load_state_dict(torch.load(directory + 'soft_q_net1.pth'))
        self.soft_q_net2.load_state_dict(torch.load(directory + 'soft_q_net2.pth'))
        self.target_soft_q_net1.load_state_dict(torch.load(directory + 'target_soft_q_net1.pth'))
        self.target_soft_q_net2.load_state_dict(torch.load(directory + 'target_soft_q_net2.pth'))
        self.policy_net.load_state_dict(torch.load(directory + 'policy_net.pth'))
        #self.critic_2_target.load_state_dict(torch.load(directory + 'critic_2_target.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")

    def UnetTrain(self):
        criterion = torch.nn.BCELoss()
        # 梯度下降
        optimizer = optim.Adam(self.unet.parameters())
        for _ in range(0, args.epoch):
            optimizer.zero_grad()
            states, labels = self.um.sample(args.batch_size_unet)
            inputs = torch.FloatTensor(states).to(device)
            labels = torch.FloatTensor(labels).to(device)
            outputs = self.unet(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            self.writer.add_scalar('Loss/U_net_loss', loss, global_step=self.unet_update_iteration)
            self.unet_update_iteration += 1
            loss.backward()  # 梯度下降,计算出梯度
            optimizer.step()

    def save_unet(self):
        torch.save(self.unet.state_dict(), directory + 'unet.pth')
        print("====================================")
        print("Unet has been saved...")
        print("====================================")

    def load_unet(self):
        self.unet.load_state_dict(torch.load(directory + 'unet.pth'))
        print("====================================")
        print("Unet has been loaded...")
        print("====================================")


# def train(agent, steps_per_epoch=1000, epochs=100, start_steps=1000, max_ep_len=200):
#     # start tracking time
#     start_time = time.time()
#     total_rewards = []
#     avg_reward = None
#
#     # set initial values
#     o, r, d, ep_reward, ep_len, ep_num = env.reset(), 0, False, 0, 0, 1
#
#     # track total steps
#     total_steps = steps_per_epoch * epochs
#
#     for t in range(1, total_steps):
#
#         explore = t < start_steps
#         a = agent.get_action(o, explore=explore)
#
#         # Step the env
#         o2, r, d, _ = env.step(a)
#         ep_reward += r
#         ep_len += 1
#
#         # Ignore the "done" signal if it comes from hitting the time
#         # horizon (that is, when it's an artificial terminal signal
#         # that isn't based on the agent's state)
#         d = False if ep_len == max_ep_len else d
#
#         # Store experience to replay buffer
#         replay_buffer.push(o, a, r, o2, d)
#
#         # Super critical, easy to overlook step: make sure to update
#         # most recent observation!
#         o = o2
#
#         if d or (ep_len == max_ep_len):
#
#             # carry out update for each step experienced (episode length)
#             if not explore:
#                 agent.update(ep_len)
#
#             # log progress
#             total_rewards.append(ep_reward)
#             avg_reward = np.mean(total_rewards[-100:])
#
#             print("Steps:{} Episode:{} Reward:{} Avg Reward:{}".format(t, ep_num, ep_reward, avg_reward))
#             #             logger.store(LossPi=outs[0], LossQ1=outs[1], LossQ2=outs[2],
#             #                          LossV=outs[3], Q1Vals=outs[4], Q2Vals=outs[5],
#             #                          VVals=outs[6], LogPi=outs[7])
#
#             #             logger.store(EpRet=ep_ret, EpLen=ep_len)
#             o, r, d, ep_reward, ep_len = env.reset(), 0, False, 0, 0
#             ep_num += 1
#
#         # End of epoch wrap-up
#         if t > 0 and t % steps_per_epoch == 0:
#             epoch = t // steps_per_epoch
#
#             # TODO: Save Model
#
#             # TODO: Test
#
#             # TODO: Log Epoch Results
def main():
    #agent = TD3(state_dim, action_dim, max_action)
    agent = SAC(env,  state_dim, action_dim)
    lv_0 = 0
    lv_1 = 0
    lv_2 = 0
    lv_3 = 0
    lv_4 = 0
    lv_5 = 0
    ep_r = 0
    pointer = 0
    if args.mode == 'test':
        agent.load(directory)
        #agent.load_unet()
        epsilon_model = 'test'
        for epoch in range(args.test_episode):  # 50
            s = env.reset(env_switch=False)
            # state = torch.FloatTensor(s).to(device)
            # state = agent.unet(state).cpu().data.numpy()
            state = s
            for t in count():
                action, noise, action_truth = agent.select_action(state, args.capacity, t, epsilon_model)
                print("Action: {} noise: {} Action After Noise: {} pointer: {}".format(action_truth, noise, action,
                                                                                       t))
                s_, reward, done, info, labdepth, lv_end = env.step(action)
                # next_state = torch.FloatTensor(s_).to(device)
                # next_state = agent.unet(next_state).cpu().data.numpy()

                next_state = s_

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
                    else:
                        lv_5 = lv_5 + 1
                        print("lv: %r" % (lv_end))
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(epoch, ep_r, t))
                    ep_r = 0
                    break
                state = next_state
        print("lv1: {} lv2: {} lv3: {} lv4: {} lv5: {}".format(lv_1, lv_2, lv_3,lv_4,lv_5))

    elif args.mode == 'train':
        print("====================================")
        print("Collection Experience...")
        print("====================================")
        if args.load: agent.load(best_dir)
        #if args.unet_load: agent.load_unet()
        if args.buffer_load: agent.memory = agent.memory.load(dir)
        epsilon_model = 'test'
        count_epoch = 0
        sum_r = []
        mean_r = -3
        for epoch in range(args.num_episode):
            if pointer <= args.capacity - 1 :
                agent.memory.save(dir)
            if len(agent.memory.sucBuffer) + len(agent.memory.buffer) >= args.wait - 1:
                replay_max = args.capacity
                replay_len = len(agent.memory.sucBuffer) + len(agent.memory.buffer)

                k = 1 + replay_len / replay_max

                batch_size = int(k * args.batch_size)
                update_times = int(k * args.epoch)
                # batch_size = args.batch_size
                # update_times = args.epoch
                agent.update(update_times, batch_size)

                print("learn")
            # if len(agent.um.buffer) >= 32:
            #     agent.UnetTrain()
            count_epoch = epoch
            if count_epoch % 10 == 0 and count_epoch != 0:
                env_switch = True
            else:
                env_switch = False
            s = env.reset(env_switch)
            # state = np.stack([s] * args.seqsize, axis=2)
            state = s
            # state = torch.FloatTensor(s).to(device)
            # state = agent.unet(state).cpu().data.numpy()
            # if pointer % 1000 == 0:
            #     dir_in = 'input' + str(epoch) + '.png'
            #     dir_out = 'output' + str(epoch) + '.png'
            #     in_map = np.array(s * 255, dtype=np.uint8)
            #     out_map = np.array(state * 255, dtype=np.uint8)
            #     cv2.imwrite("C:\\Unet_img\\"+dir_in, in_map[0][0])
            #     cv2.imwrite("C:\\Unet_img\\" + dir_out, out_map[0][0])
            ep_r = 0

            for t in range(args.max_frame):

                action, noise, action_truth = agent.select_action(state, args.capacity, pointer, epsilon_model)
                #action = action + np.random.normal(0, args.exploration_noise, size=action_dim)
                #action = action.clip(-max_action, max_action)
                print("Action: {}  pointer: {} step: {}".format(action_truth, pointer, t))
                s_, reward, done, info, labdepth, level_end = env.step(action)

                if info == "through":
                    s_[:] = 0
                    print("camera go through the wall!!!")
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
                #agent.um.append(s_, labdepth)
                # next_state = np.append(state[:, :, 1:], [s_], axis=2)
                # next_state = torch.FloatTensor(s_).to(device)
                # next_state = agent.unet(next_state).cpu().data.numpy()
                next_state = s_

                # unetshow = np.array(next_state * 255, dtype=np.uint8)
                # cv2.imshow('Unet_Out', unetshow[0][0])
                # k = cv2.waitKey(1) & 0xff
                ep_r += reward
                # if args.render and epoch >= args.render_interval:
                #     env.render()
                #agent.memory.push((state, next_state, action, reward, np.float(done)))
                index = [0]

                new_a = np.delete(action, index)
                #agent.memory.push(state, new_a, reward, next_state, np.float(done))
                agent.memory.append(state, new_a, reward, next_state, np.float(done), info)
                pointer += 1
                state = next_state

                #learn



                if done or t == args.max_frame - 1:
                    agent.writer.add_scalar('ep_r', ep_r, global_step=epoch)
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

                #agent.save_unet()


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