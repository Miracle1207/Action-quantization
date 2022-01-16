import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
import torch.distributions as D

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)
'''
GMM: MixtureSameFamily
'''
class GaussianMixturePolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, num_k, action_space=None):
        super(GaussianMixturePolicy, self).__init__()
        self.num_k = num_k
        self.num_actions = num_actions
        self.k_means_linear = []
        self.k_stds_linear = []
        # self.k_weight_linear = []

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        for i in range(num_k):
            self.k_means_linear.append(nn.Linear(hidden_dim, num_actions))
            self.k_stds_linear.append(nn.Linear(hidden_dim, num_actions))
            # self.k_weight_linear.append(nn.Linear(hidden_dim, 1))

        self.k_weight_linear = nn.Linear(hidden_dim, num_k)
        # TODO: weights update?
        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        k_means = []
        k_stds = []
        # k_weight = []
        self.d = 1
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        # contant_x = F.relu(self.linear1(torch.ones(state.shape)))
        # contant_x = F.relu(self.linear2(contant_x))

        for k_i in range(self.num_k):
            if k_i == 0:
                k_means = torch.unsqueeze(self.k_means_linear[k_i](x), dim=self.d)
                log_std = self.k_stds_linear[k_i](x)
                k_log_stds = torch.unsqueeze(torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX).exp(), dim=self.d)  # 将log_std限制到一个空间内
                # k_weight = self.k_weight_linear[k_i](x)

            else:
                k_means = torch.cat((k_means, torch.unsqueeze(self.k_means_linear[k_i](x), dim=self.d)), dim=self.d)
                log_std = self.k_stds_linear[k_i](x)
                k_log_stds = torch.cat((k_log_stds, torch.unsqueeze(torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX).exp(), dim=self.d)), dim=self.d)
                # k_weight = torch.cat([k_weight, self.k_weight_linear[k_i](x)], dim=0)

        k_weight = self.k_weight_linear(x)
        k_w = abs(k_weight)

        return k_means, k_log_stds, k_w
        # return k_means, k_w

    def sample(self, state):
        mean, std, k_w = self.forward(state)
        mix = D.Categorical(k_w)
        comp = D.Independent(D.Normal(mean, std), 1)
        if mix.logits.shape[-1] != comp.batch_shape[-1]:
            print("wrong")
        gmm = D.MixtureSameFamily(mix, comp)
        x_t = gmm.sample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = gmm.log_prob(x_t)
        # Enforcing Action Bound
        log_prob = log_prob[0] - torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.mean(mean, dim=self.d)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob.unsqueeze(0), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianMixturePolicy, self).to(device)
'''
without variance
'''
# class GaussianMixturePolicy(nn.Module):
#     def __init__(self, num_inputs, num_actions, hidden_dim, num_k, action_space=None):
#         super(GaussianMixturePolicy, self).__init__()
#         self.num_k = num_k
#         self.num_actions = num_actions
#         self.k_means_linear = []
#         self.k_stds_linear = []
#
#         self.linear1 = nn.Linear(num_inputs, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, hidden_dim)
#
#         for i in range(num_k):
#             self.k_means_linear.append(nn.Linear(hidden_dim, num_actions))
#             # self.k_stds_linear.append(nn.Linear(hidden_dim, num_actions))
#
#         self.k_weight_linear = nn.Linear(hidden_dim, num_k)
#
#         # TODO: weights update?
#         self.apply(weights_init_)
#
#         # action rescaling
#         if action_space is None:
#             self.action_scale = torch.tensor(1.)
#             self.action_bias = torch.tensor(0.)
#         else:
#             self.action_scale = torch.FloatTensor(
#                 (action_space.high - action_space.low) / 2.)
#             self.action_bias = torch.FloatTensor(
#                 (action_space.high + action_space.low) / 2.)
#
#     def forward(self, state):
#         k_means = []
#         k_log_stds = []
#
#         x = F.relu(self.linear1(state))
#         x = F.relu(self.linear2(x))
#
#         for k_i in range(self.num_k):
#             k_means.append(self.k_means_linear[k_i](x))
#             # log_std = self.k_stds_linear[k_i](x)
#             # k_log_stds.append(torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX))  # 将log_std限制到一个空间内
#
#         k_weight = self.k_weight_linear(x)
#         k_w = abs(k_weight[0].detach().numpy())
#         k_w = k_w/sum(k_w)
#         # return torch(k_means), k_log_stds, k_w
#         return k_means, k_w
#
#     def sample(self, state):
#         normal = []
#         gmm = 0
#         # k_means, k_log_stds, k_w = self.forward(state)
#         k_means, k_w = self.forward(state)
#         # for k_i in range(self.num_k):
#         #     std = k_log_stds[k_i].exp()
#         #     normal.append(Normal(k_means[k_i], std))
#         #     gmm += k_w[k_i]*Normal(k_means[k_i], std)
#         index = np.random.choice(a=list(range(self.num_k)), p=k_w)
#         x_t = k_means[index]
#         # x_t = gmm.rsample()  # for reparameterization trick (mean + std * N(0,1))
#         y_t = torch.tanh(x_t)
#         action = y_t * self.action_scale + self.action_bias
#
#         for k_i in range(self.num_k):
#             k_w[k_i] = np.log(k_w[k_i])
#         k_w = np.expand_dims(k_w, axis=0)
#         log_prob = torch.from_numpy(k_w)
#
#         # Enforcing Action Bound
#         # log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
#         log_prob = log_prob.sum(1, keepdim=True)
#         # mean = torch.tanh(mean) * self.action_scale + self.action_bias
#         return action, log_prob, action

'''
b_k without s_t, but on rou_k(a,s), w_k(s), sigma_k(s)
'''
# class GaussianMixturePolicy(nn.Module):
#     def __init__(self, num_inputs, num_actions, hidden_dim, num_k, action_space=None):
#         super(GaussianMixturePolicy, self).__init__()
#         self.num_k = num_k
#         self.num_actions = num_actions
#         self.k_stds_linear = []
#
#         self.linear1 = nn.Linear(num_inputs, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, hidden_dim)
#
#         for i in range(num_k):
#             self.k_stds_linear.append(nn.Linear(hidden_dim, num_actions))
#
#         self.k_weight_linear = nn.Linear(hidden_dim, num_k)
#
#         # TODO: weights update?
#         self.apply(weights_init_)
#
#         # init means
#         action_index = np.array(range(num_k)).reshape(num_k, 1)
#         action_interval = (action_space.high - action_space.low).reshape(1, action_space.shape[0]) / num_k
#         self.k_means = torch.from_numpy(np.dot(action_index, action_interval) + action_space.low)
#
#         # action rescaling
#         if action_space is None:
#             self.action_scale = torch.tensor(1.)
#             self.action_bias = torch.tensor(0.)
#         else:
#             self.action_scale = torch.FloatTensor(
#                 (action_space.high - action_space.low) / 2.)
#             self.action_bias = torch.FloatTensor(
#                 (action_space.high + action_space.low) / 2.)
#
#     def forward(self, state):
#         k_log_stds = []
#
#         x = F.relu(self.linear1(state))
#         x = F.relu(self.linear2(x))
#
#         for k_i in range(self.num_k):
#             log_std = self.k_stds_linear[k_i](x)
#             k_log_stds.append(torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX))  # 将log_std限制到一个空间内
#
#         k_weight = self.k_weight_linear(x)
#         return k_log_stds, k_weight
#
#     # 高斯分布概率
#     def gaussian_prob(self, data, avg, sig):
#         sqrt_2pi = np.power(2 * np.pi, 0.5)
#         coef = 1 / (sqrt_2pi * sig)
#         powercoef = -1 / (2 * np.power(sig, 2))
#         mypow = powercoef * (np.power((data - avg), 2))
#         return coef * (np.exp(mypow))
#
#     '''
#     batch_action is continuous action, sampled by N(a|mu,sigma), todo: push into replay buffer
#     '''
#     def update_bk(self, batch_state, batch_action):
#         # k_w to numpy
#         kws = abs(kws[0].detach().numpy())
#         kws = kws / sum(kws)
#         # k_log_stds to numpy
#         log_stds = log_stds.detach().numpy()
#         sigma = np.diag(lo)
#
#         action = np.random.multivariate_normal(mean=, cov= , size=)
#
#         # p(b|a,s) todo: update parameter
#         rou_k = k_w * self.gaussian_prob(, self.k_means, k_log_stds)/\
#                 sum(k_w * self.gaussian_prob(self.k_means, k_log_stds))
#
#         # sample b_t according to k_w
#
#         k_means =
#         return k_means
#
#
#
#     def sample(self, state):
#         k_log_stds, k_w = self.forward(state)
#         index = np.random.choice(a=list(range(self.num_k)), p=k_w)
#         x_t = self.k_means[index]
#         # x_t = gmm.rsample()  # for reparameterization trick (mean + std * N(0,1))
#         y_t = torch.tanh(x_t)
#         action = y_t * self.action_scale + self.action_bias
#
#         for k_i in range(self.num_k):
#             k_w[k_i] = np.log(k_w[k_i])
#         k_w = np.expand_dims(k_w, axis=0)
#         log_prob = torch.from_numpy(k_w)
#
#         log_prob = log_prob.sum(1, keepdim=True)
#         mean = self.k_means/sum(self.k_means)
#         mean = torch.tanh(mean) * self.action_scale + self.action_bias
#         return action, log_prob, mean

'''
bk without state SAC+GMM
'''
# class GaussianMixturePolicy(nn.Module):
#     def __init__(self, num_inputs, num_actions, hidden_dim, num_k, action_space=None):
#         super(GaussianMixturePolicy, self).__init__()
#         self.num_k = num_k
#         self.num_actions = num_actions
#         self.k_means_linear = []
#         self.k_stds_linear = []
#
#         self.linear1 = nn.Linear(num_inputs, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, hidden_dim)
#
#         for i in range(num_k):
#             self.k_means_linear.append(nn.Linear(hidden_dim, num_actions))
#             # self.k_stds_linear.append(nn.Linear(hidden_dim, num_actions))
#
#         self.k_weight_linear = nn.Linear(hidden_dim, num_k)
#
#         # TODO: weights update?
#         self.apply(weights_init_)
#
#         # action rescaling
#         if action_space is None:
#             self.action_scale = torch.tensor(1.)
#             self.action_bias = torch.tensor(0.)
#         else:
#             self.action_scale = torch.FloatTensor(
#                 (action_space.high - action_space.low) / 2.)
#             self.action_bias = torch.FloatTensor(
#                 (action_space.high + action_space.low) / 2.)
#
#     def forward(self, state):
#         k_means = []
#         k_log_stds = []
#
#         x = F.relu(self.linear1(state))
#         x = F.relu(self.linear2(x))
#         contant_x = F.relu(self.linear1(torch.ones(state.shape)))
#         contant_x = F.relu(self.linear2(contant_x))
#
#         for k_i in range(self.num_k):
#             k_means.append(self.k_means_linear[k_i](contant_x))
#
#         k_weight = self.k_weight_linear(x)
#         k_w = abs(k_weight[0].detach().numpy())
#         k_w = k_w/sum(k_w)
#         # return torch(k_means), k_log_stds, k_w
#         return k_means, k_w
#
#     def sample(self, state):
#         normal = []
#         gmm = 0
#
#         k_means, k_w = self.forward(state)
#
#         index = np.random.choice(a=list(range(self.num_k)), p=k_w)
#         x_t = k_means[index]
#         # x_t = gmm.rsample()  # for reparameterization trick (mean + std * N(0,1))
#         y_t = torch.tanh(x_t)
#         action = y_t * self.action_scale + self.action_bias
#
#         for k_i in range(self.num_k):
#             k_w[k_i] = np.log(k_w[k_i])
#         k_w = np.expand_dims(k_w, axis=0)
#         log_prob = torch.from_numpy(k_w)
#
#         # Enforcing Action Bound
#         # log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
#         log_prob = log_prob.sum(1, keepdim=True)
#         # mean = torch.tanh(mean) * self.action_scale + self.action_bias
#         return action, log_prob, action


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
