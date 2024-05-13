import torch
import torch.nn as nn
import torch.distributions as distributions

HIDDEN1 = 400
HIDDEN2 = 300
HIDDEN3 = 300

class Actor(nn.Module):
    def __init__(self, name, trainable=True, input_size=29):
        super(Actor, self).__init__()
        self.trainable = trainable
        self.name = name
        self.input_size = input_size

        # Initializers
        self.xavier = nn.init.xavier_normal_
        self.bias_const = nn.init.constant_
        self.rand_unif = nn.init.uniform_

        # Regularizer
        self.regularizer = nn.MSELoss()  # L2 regularization can be handled during the optimization step in PyTorch

        # Layers
        self.l1 = nn.Linear(self.input_size, HIDDEN1)
        self.l2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.l3 = nn.Linear(HIDDEN2, HIDDEN3)

        self.mu_st = nn.Linear(HIDDEN3, 1)
        self.mu_acc = nn.Linear(HIDDEN3, 1)
        self.mu_br = nn.Linear(HIDDEN3, 1)

        self.sigma_st = nn.Linear(HIDDEN3, 1)
        self.sigma_acc = nn.Linear(HIDDEN3, 1)
        self.sigma_br = nn.Linear(HIDDEN3, 1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, tfs):
        l1 = self.relu(self.l1(tfs))
        l2 = self.relu(self.l2(l1))
        l3 = self.relu(self.l3(l2))

        mu_st = self.tanh(self.mu_st(l3))
        mu_acc = self.sigmoid(self.mu_acc(l3))
        mu_br = self.sigmoid(self.mu_br(l3)) * 0.1  # scalar multiplication

        sigma_st = self.sigmoid(self.sigma_st(l3)) * 0.3  # scalar multiplication
        sigma_acc = self.sigmoid(self.sigma_acc(l3)) * 0.5  # scalar multiplication
        sigma_br = self.sigmoid(self.sigma_br(l3)) * 0.05  # scalar multiplication

        mu = torch.cat([mu_st, mu_acc, mu_br], dim=1)
        sigma = torch.cat([sigma_st, sigma_acc, sigma_br], dim=1)

        norm_dist = distributions.Normal(mu, sigma)
        return norm_dist