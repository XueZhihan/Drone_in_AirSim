import argparse
from collections import namedtuple
from itertools import count
from algorithms.ReplayMemory import ReplayMemory
import os, sys, random
import numpy as np
from drone_env import drone_env_block
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter
import math
from OU import OU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='test', type=str)  # mode = 'train' or 'test'
parser.add_argument("--env_name",
                    default="Drone1")  # OpenAI gym environment name， BipedalWalker-v2  Pendulum-v0
parser.add_argument('--tau', default=2.5e-3, type=float)  # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_episode', default=50, type=int)
parser.add_argument('--epoch', default=10, type=int)  # buffer采样的数据训练几次
parser.add_argument('--learning_rate', default=3e-8, type=float)
parser.add_argument('--gamma', default=0.99, type=int)  # discounted factor
parser.add_argument('--capacity', default=3000, type=int)  # replay buffer size
parser.add_argument('--num_episode', default=20000, type=int)  # num of episodes in training
parser.add_argument('--batch_size', default=32, type=int)  # mini batch size
parser.add_argument('--seed', default=True, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)

# optional parameters
# parser.add_argument('--num_hidden_layers', default=2, type=int)
# parser.add_argument('--sample_frequency', default=256, type=int)
# parser.add_argument('--activation', default='Relu', type=str)
parser.add_argument('--render', default=False, type=bool)  # show UI or not
parser.add_argument('--log_interval', default=10, type=int)  # 每10episode保存一次模型
parser.add_argument('--load', default=True, type=bool)  # 训练前是否读取模型
parser.add_argument('--render_interval', default=100, type=int)  # after render_interval, the env.render() will work
parser.add_argument('--policy_noise', default=0.2, type=float)  # 动作向量的噪声扰动的方差
parser.add_argument('--noise_clip', default=0.5, type=float)
parser.add_argument('--policy_delay', default=2, type=int)
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_frame', default=2000, type=int)
parser.add_argument('--print_log', default=1, type=int)
parser.add_argument('--seqsize', type=int, default=8)

parser.add_argument('--actor_lr',   type=float, default=5e-5)
parser.add_argument('--critic_lr',  type=float, default=2.5e-4)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)
env = drone_env_block(aim=[58, 125, 10])
# env = env.unwrapped
#
# if args.seed:
#     env.seed(args.random_seed)
#     torch.manual_seed(args.random_seed)
#     np.random.seed(args.random_seed)
OU = OU()
train_indicator = 1

state_dim = 2880#[None, 1, 8, 144, 256]
action_dim = 3
max_action = 1  # 动作取值上界5tr76
min_Val = torch.tensor(1e-7).float().to(device)  # min value

directory = './exp' + script_name + args.env_name + './'
'''
Implementation of TD3 with pytorch 
Original paper: https://arxiv.org/abs/1802.09477
Not the author's implementation !
'''


class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''

    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

class Feature(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Feature, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(5, 5, 3), stride=1, padding=1)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        #self.gru = nn.GRU(2880, 200)
        self.norm1 = nn.BatchNorm3d(1)
        self.norm2 = nn.BatchNorm3d(64)

    def forward(self, state):
        x = self.norm1(state)
        x = F.max_pool3d(F.relu(self.conv1(x)), (3, 3, 3))

        x = F.max_pool3d(F.relu(self.conv2(x)), (2, 3, 3))
        x = F.max_pool3d(F.relu(self.conv3(x)), (1, 3, 3))
        x = self.norm2(x)
        fe = x.view(-1, self.num_flat_features(x))


        return fe

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1

        for s in size:
            num_features *= s

        return num_features
class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(5, 5, 3), stride=1, padding=1)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        #self.gru = nn.GRU(2880, 200)
        self.norm1 = nn.BatchNorm3d(1)
        self.norm2 = nn.BatchNorm3d(64)
        self.fc1 = nn.Linear(2880, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, action_dim)

        self.max_action = max_action

    def forward(self, state):
        x = self.norm1(state)
        x = F.max_pool3d(F.relu(self.conv1(x)), (3, 3, 3))

        x = F.max_pool3d(F.relu(self.conv2(x)), (2, 3, 3))
        x = F.max_pool3d(F.relu(self.conv3(x)), (1, 3, 3))
        x = self.norm2(x)
        f = x.view(-1, self.num_flat_features(x))

        #x = self.gru(x, (5, 32, 200))
        a = F.relu(self.fc1(f))
        a = F.relu(self.fc2(a))
        a = torch.tanh(self.fc3(a)) * self.max_action
        return [f, a]

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1

        for s in size:
            num_features *= s

        return num_features

class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # self.conv1 = nn.Conv3d(1, 64, kernel_size=(5, 5, 3), stride=1, padding=1)
        # self.conv2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        # self.conv3 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        # #self.gru = nn.GRU(2880, 200)
        # self.norm1 = nn.BatchNorm3d(1)
        # self.norm2 = nn.BatchNorm3d(64)
        self.fc1 = nn.Linear(2880+action_dim, 200+action_dim)
        self.fc2 = nn.Linear(200+action_dim, 200)
        self.fc3 = nn.Linear(200, 1)

    def forward(self, state, action):
        # x = self.norm1(state)
        # x = F.max_pool3d(F.relu(self.conv1(x)), (3, 3, 3))
        #
        # x = F.max_pool3d(F.relu(self.conv2(x)), (2, 3, 3))
        # x = F.max_pool3d(F.relu(self.conv3(x)), (1, 3, 3))
        # x = self.norm2(x)
        # x = x.view(-1, self.num_flat_features(x))
        #x = self.gru(x, (5, 32, 200))
        state_action = torch.cat([state, action], 1)
        q = F.relu(self.fc1(state_action))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1

        for s in size:
            num_features *= s

        return num_features

# class Critic2(nn.Module):
#
#     def __init__(self, state_dim, action_dim):
#         super(Critic2, self).__init__()
#         # self.conv1 = nn.Conv3d(1, 64, kernel_size=(5, 5, 3), stride=1, padding=1)
#         # self.conv2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
#         # self.conv3 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
#         # #self.gru = nn.GRU(2880, 200)
#         # self.norm1 = nn.BatchNorm3d(1)
#         # self.norm2 = nn.BatchNorm3d(64)
#         self.fc1 = nn.Linear(2880+action_dim, 200+action_dim)
#         self.fc2 = nn.Linear(200+action_dim, 200)
#         self.fc3 = nn.Linear(200, 1)
#
#     def forward(self, state, action):
#         # x = self.norm1(state)
#         # x = F.max_pool3d(F.relu(self.conv1(x)), (3, 3, 3))
#         #
#         # x = F.max_pool3d(F.relu(self.conv2(x)), (2, 3, 3))
#         # x = F.max_pool3d(F.relu(self.conv3(x)), (1, 3, 3))
#         # x = self.norm2(x)
#         # x = x.view(-1, self.num_flat_features(x))
#         #x = self.gru(x, (5, 32, 200))
#         state_action = torch.cat([state, action], 1)
#         q = F.relu(self.fc1(state_action))
#         q = F.relu(self.fc2(q))
#         q = self.fc3(q)
#         return q
#
#     def num_flat_features(self, x):
#         size = x.size()[1:]  # all dimensions except the batch dimension
#         num_features = 1
#
#         for s in size:
#             num_features *= s
#
#         return num_features

class TD3():
    def __init__(self, state_dim, action_dim, max_action):
        self.lr_actor = args.actor_lr
        self.lr_critic = args.critic_lr
        self.betas = (0.9, 0.999)
        # 6个网络
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor = torch.nn.DataParallel(self.actor)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = torch.nn.DataParallel(self.actor_target)
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1 = torch.nn.DataParallel(self.critic_1)
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = torch.nn.DataParallel(self.critic_1_target)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2 = torch.nn.DataParallel(self.critic_2)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = torch.nn.DataParallel(self.critic_2_target)
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor, betas=self.betas)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=self.lr_critic, betas=self.betas)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=self.lr_critic, betas=self.betas)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.max_action = max_action
        #self.memory = Replay_buffer(args.capacity)
        self.memory = ReplayMemory(args.capacity)
        self.writer = SummaryWriter(directory)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def  select_action(self, state, wait_before_train, step, epsilon_model):
        epsilon = 1
        noise = True
        state = torch.tensor(state).float().to(device)
        action = self.actor(state)[1].cpu().data.numpy().flatten()
        action_truth = np.zeros([3])
        noise_t = np.zeros([action_dim])
        epsilon_ceil = 0.95
        if epsilon_model == 'test':
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

            noise_t[0] =train_indicator * OU.function(action[0], 0.5, 0.6, 0.10)
            noise_t[1] = train_indicator * OU.function(action[1], 0.0, 0.6, 0.30)
            noise_t[2] = train_indicator * OU.function(action[2], 0.0, 0.6, 0.30)

            action[0] = np.clip(action[0]+noise_t[0], -self.max_action, self.max_action)
            action[1] = np.clip(action[1] + noise_t[1], -self.max_action, self.max_action)
            action[2] = np.clip(action[2] + noise_t[2], -self.max_action, self.max_action)


        else:
            # for i in range(3):
            #     action[i] = np.clip(action[i], -self.a_bound, self.a_bound)
            action_truth[0] = action[0]
            action_truth[1] = action[1]
            action_truth[2] = action[2]

            # noise_t[0] =train_indicator * OU.function(action[0], 0.5, 0.06, 0.10)
            # noise_t[0] = train_indicator * OU.function(action[1], 0.0, 0.06, 0.30)
            # noise_t[1] = train_indicator * OU.function(action[2], 0.0, 0.06, 0.30)

            action[0] = np.clip(action[0]+noise_t[0], -self.max_action, self.max_action)
            action[1] = np.clip(action[1] + noise_t[1], -self.max_action, self.max_action)
            action[2] = np.clip(action[2] + noise_t[2], -self.max_action, self.max_action)
        return action, noise_t, action_truth

    def update(self, epoch):

        # if self.num_training % 500 == 0:
        # print("====================================")
        # print("model has been trained for {} times...".format(self.num_training))
        # print("====================================")
        for i in range(epoch):
            x, u, r, y, d = self.memory.sample(args.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Select next action according to target policy:
            noise = torch.ones_like(action).data.normal_(0, args.policy_noise).to(device)
            noise = noise.clamp(-args.noise_clip, args.noise_clip)
            next_action = (self.actor_target(next_state)[1] + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute target Q-value:
            next_state_f = self.actor(next_state)[0]
            target_Q1 = self.critic_1_target(next_state_f, next_action)
            target_Q2 = self.critic_2_target(next_state_f, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            #target_Q = target_Q1.min(target_Q2)
            target_Q = target_Q.flatten()
            target_Q = reward + ((1 - done) * args.gamma * target_Q).detach()

            # Optimize Critic 1:
            state_f = self.actor(state)[0]
            current_Q1 = self.critic_1(state_f, action)
            current_Q1 = current_Q1.flatten()
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward(retain_graph=True)
            self.critic_1_optimizer.step()
            self.writer.add_scalar('Loss/Q1_loss', loss_Q1, global_step=self.num_critic_update_iteration)

            # Optimize Critic 2:
            current_Q2 = self.critic_2(state_f, action)
            current_Q2 = current_Q2.flatten()
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward(retain_graph=True)
            self.critic_2_optimizer.step()
            self.writer.add_scalar('Loss/Q2_loss', loss_Q2, global_step=self.num_critic_update_iteration)
            # Delayed policy updates:
            if i % args.policy_delay == 0:
                # Compute actor loss:
                actor_loss = - self.critic_1(state_f, self.actor(
                    state)[1]).mean()  # 随着更新的进行Q1和Q2两个网络，将会变得越来越像。所以用Q1还是Q2，还是两者都用，对于actor的问题不大。

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(((1 - args.tau) * target_param.data) + args.tau * param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(((1 - args.tau) * target_param.data) + args.tau * param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(((1 - args.tau) * target_param.data) + args.tau * param.data)

                self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1
        self.num_training += 1

    def save(self):
        torch.save(self.actor.state_dict(), directory + 'actor.pth')
        torch.save(self.actor_target.state_dict(), directory + 'actor_target.pth')
        torch.save(self.critic_1.state_dict(), directory + 'critic_1.pth')
        torch.save(self.critic_1_target.state_dict(), directory + 'critic_1_target.pth')
        torch.save(self.critic_2.state_dict(), directory + 'critic_2.pth')
        torch.save(self.critic_2_target.state_dict(), directory + 'critic_2_target.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.actor_target.load_state_dict(torch.load(directory + 'actor_target.pth'))
        self.critic_1.load_state_dict(torch.load(directory + 'critic_1.pth'))
        self.critic_1_target.load_state_dict(torch.load(directory + 'critic_1_target.pth'))
        self.critic_2.load_state_dict(torch.load(directory + 'critic_2.pth'))
        self.critic_2_target.load_state_dict(torch.load(directory + 'critic_2_target.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


def main():
    agent = TD3(state_dim, action_dim, max_action)
    ep_r = 0
    pointer = 0
    if args.mode == 'test':
        agent.load()
        epsilon_model = 'test'
        for epoch in range(args.test_episode):  # 50
            s = env.reset()
            state = np.stack([s] * args.seqsize, axis=2)
            for t in count():
                action, noise, action_truth = agent.select_action(state, args.capacity, t, epsilon_model)
                print("Action: {} noise: {} Action After Noise: {} pointer: {}".format(action_truth, noise, action,
                                                                                       t))
                s_, reward, done, info = env.step(action)
                next_state = np.append(state[:, :, 1:], [s_], axis=2)
                ep_r += reward
                #env.render()
                if done or t == args.max_frame - 1:
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(epoch, ep_r, t))
                    break
                state = next_state

    elif args.mode == 'train':
        print("====================================")
        print("Collection Experience...")
        print("====================================")
        if args.load: agent.load()
        epsilon_model = 'linear'
        for epoch in range(args.num_episode):
            s = env.reset()
            state = np.stack([s] * args.seqsize, axis=2)
            if len(agent.memory.buffer) >= args.capacity - 1:
                agent.update(args.epoch)
            for t in range(args.max_frame):
                action, noise, action_truth = agent.select_action(state, args.capacity, pointer, epsilon_model)
                #action = action + np.random.normal(0, args.exploration_noise, size=action_dim)
                #action = action.clip(-max_action, max_action)
                print("Action: {} noise: {} Action After Noise: {} pointer: {} step: {}".format(action_truth, noise, action,
                                                                                       pointer,t))
                s_, reward, done, info = env.step(action)
                next_state = np.append(state[:, :, 1:], [s_], axis=2)
                ep_r += reward
                # if args.render and epoch >= args.render_interval:
                #     env.render()
                #agent.memory.push((state, next_state, action, reward, np.float(done)))
                agent.memory.append(state, action, reward, next_state, np.float(done))
                pointer += 1
                state = next_state

                #learn

                if done or t == args.max_frame - 1:
                    agent.writer.add_scalar('ep_r', ep_r, global_step=epoch)
                    if epoch % args.print_log == 0:
                        print("Ep_i {}, the ep_r is {:0.2f}, the step is {}".format(epoch, ep_r, t))
                    ep_r = 0
                    break
            if epoch % args.log_interval == 0:
                agent.save()

    else:
        raise NameError("mode wrong!!!")


if __name__ == '__main__':
    main()