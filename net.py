import torch
import torch.nn as nn
import numpy as np

'''Q Network'''




class Actor(nn.Module):  # DPG: Deterministic Policy Gradient
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, action_dim))

    def forward(self, state):
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state, action_std):
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)

class Critic(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, 1))

    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim=1))  # Q value

class ActorSAC(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
        super().__init__()
        if if_use_dn:  # use a DenseNet (DenseNet has both shallow and deep linear layer)
            nn_dense_net = DenseNet(mid_dim)
            self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                           nn_dense_net, )
            lay_dim = nn_dense_net.out_dim
        else:  # use a simple network. Deeper network does not mean better performance in RL.
            lay_dim = mid_dim
            self.lstm = nn.LSTM(state_dim, 8, batch_first=True)
            self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                           nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                           nn.Linear(mid_dim, lay_dim), nn.Hardswish())
        self.net_a_avg = nn.Linear(lay_dim, action_dim)  # the average of action
        self.net_a_std = nn.Linear(lay_dim, action_dim)  # the log_std of action

        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
        layer_norm(self.net_a_avg, std=0.01)  # output layer for action, it is no necessary.

    def forward(self, state):
        state = self.lstm(state)
        tmp = self.net_state(state)
        return self.net_a_avg(tmp).tanh()  # action

    def get_action(self, state):
        state = self.lstm(state)[0]
        state = torch.flatten(state,start_dim=1)
        t_tmp = self.net_state(state)
        a_avg = self.net_a_avg(t_tmp)  # NOTICE! it is a_avg without .tanh()
        a_std = self.net_a_std(t_tmp).clamp(-20, 2).exp()
        return torch.normal(a_avg, a_std).tanh()  # re-parameterize

    def get_action_logprob(self, state):
        state = self.lstm(state)[0]
        state = torch.flatten(state, start_dim=1)
        t_tmp = self.net_state(state)
        a_avg = self.net_a_avg(t_tmp)  # NOTICE! it needs a_avg.tanh()
        a_std_log = self.net_a_std(t_tmp).clamp(-20, 2)
        a_std = a_std_log.exp()

        """add noise to action in stochastic policy"""
        noise = torch.randn_like(a_avg, requires_grad=True)
        action = a_avg + a_std * noise
        a_tan = action.tanh()  # action.tanh()
        # Can only use above code instead of below, because the tensor need gradients here.
        # a_noise = torch.normal(a_avg, a_std, requires_grad=True)

        '''compute logprob according to mean and std of action (stochastic policy)'''
        # # self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
        # logprob = a_std_log + self.sqrt_2pi_log + noise.pow(2).__mul__(0.5)  # noise.pow(2) * 0.5
        # different from above (gradient)
        delta = ((a_avg - action) / a_std).pow(2).__mul__(0.5)
        logprob = a_std_log + self.sqrt_2pi_log + delta
        # same as below:
        # from torch.distributions.normal import Normal
        # logprob_noise = Normal(a_avg, a_std).logprob(a_noise)
        # logprob = logprob_noise + (-a_noise_tanh.pow(2) + 1.000001).log()
        # same as below:
        # a_delta = (a_avg - a_noise).pow(2) /(2*a_std.pow(2))
        # logprob_noise = -a_delta - a_std.log() - np.log(np.sqrt(2 * np.pi))
        # logprob = logprob_noise + (-a_noise_tanh.pow(2) + 1.000001).log()

        logprob = logprob + (-a_tan.pow(2) + 1.000001).log()  # fix logprob using the derivative of action.tanh()
        # same as below:
        # epsilon = 1e-6
        # logprob = logprob_noise - (1 - a_noise_tanh.pow(2) + epsilon).log()
        return a_tan, logprob.sum(1, keepdim=True)


'''Value Network (Critic)'''



class CriticTwin(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
        super().__init__()

        if if_use_dn:  # use DenseNet (DenseNet has both shallow and deep linear layer)
            nn_dense = DenseNet(mid_dim)
            lay_dim = nn_dense.out_dim
            self.net_sa = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                        nn_dense, )  # state-action value function
        else:  # use a simple network for actor. Deeper network does not mean better performance in RL.
            lay_dim = mid_dim
            self.lstm = nn.LSTM(state_dim, 8, batch_first=True)
            self.net_sa = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, lay_dim), nn.ReLU())

        self.net_q1 = nn.Linear(lay_dim, 1)
        self.net_q2 = nn.Linear(lay_dim, 1)
        layer_norm(self.net_q1, std=0.1)
        layer_norm(self.net_q2, std=0.1)

    def forward(self, state, action):
        state = self.lstm(state)[0]
        state = torch.flatten(state, start_dim=1)
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp)  # one Q value

    def get_q1_q2(self, state, action):
        state = self.lstm(state)[0]
        state = torch.flatten(state, start_dim=1)
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp), self.net_q2(tmp)  # two Q values


'''Integrated Network (Parameter sharing)'''



"""utils"""


class NnnReshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.view((x.size(0),) + self.args)


class DenseNet(nn.Module):  # plan to hyper-param: layer_number
    def __init__(self, mid_dim):
        super().__init__()
        assert (mid_dim / (2 ** 3)) % 1 == 0

        def set_dim(i):
            return int((3 / 2) ** i * mid_dim)

        self.dense1 = nn.Sequential(nn.Linear(set_dim(0), set_dim(0) // 2), nn.Hardswish())
        self.dense2 = nn.Sequential(nn.Linear(set_dim(1), set_dim(1) // 2), nn.Hardswish())
        self.out_dim = set_dim(2)

        layer_norm(self.dense1[0], std=1.0)
        layer_norm(self.dense2[0], std=1.0)

    def forward(self, x1):
        x2 = torch.cat((x1, self.dense1(x1)), dim=1)
        x3 = torch.cat((x2, self.dense2(x2)), dim=1)
        return x3


def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


def demo_conv2d_state():
    state_dim = (4, 96, 96)
    batch_size = 3

    def set_dim(i):
        return int(12 * 1.5 ** i)

    mid_dim = 128
    net = nn.Sequential(NnnReshape(*state_dim),  # -> [batch_size, 4, 96, 96]
                        nn.Conv2d(state_dim[0], set_dim(0), 4, 2, bias=True), nn.LeakyReLU(),
                        nn.Conv2d(set_dim(0), set_dim(1), 3, 2, bias=False), nn.ReLU(),
                        nn.Conv2d(set_dim(1), set_dim(2), 3, 2, bias=False), nn.ReLU(),
                        nn.Conv2d(set_dim(2), set_dim(3), 3, 2, bias=True), nn.ReLU(),
                        nn.Conv2d(set_dim(3), set_dim(4), 3, 1, bias=True), nn.ReLU(),
                        nn.Conv2d(set_dim(4), set_dim(5), 3, 1, bias=True), nn.ReLU(),
                        NnnReshape(-1),
                        nn.Linear(set_dim(5), mid_dim), nn.ReLU())

    inp_shape = list(state_dim)
    inp_shape.insert(0, batch_size)
    inp = torch.ones(inp_shape, dtype=torch.float32)
    inp = inp.view(3, -1)
    print(inp.shape)
    out = net(inp)
    print(out.shape)
    exit()