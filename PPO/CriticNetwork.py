import torch
import torch.nn as nn

HIDDEN1 = 400
HIDDEN2 = 300
HIDDEN3 = 300

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        
        # Define layers
        self.l1 = nn.Linear(29, HIDDEN1)  # Assuming input size is 29 as per S_DIM in PPO
        self.l2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.l3 = nn.Linear(HIDDEN2, HIDDEN3)
        self.v = nn.Linear(HIDDEN3, 1)
        
        # Activation function
        self.relu = nn.ReLU()
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Xavier uniform initializer
        nn.init.xavier_uniform_(self.l1.weight)
        nn.init.constant_(self.l1.bias, 0.05)

        nn.init.xavier_uniform_(self.l2.weight)
        nn.init.constant_(self.l2.bias, 0.05)

        nn.init.xavier_uniform_(self.l3.weight)
        nn.init.constant_(self.l3.bias, 0.05)

        # Random uniform initializer for the output layer
        nn.init.uniform_(self.v.weight, -0.003, 0.003)
        nn.init.constant_(self.v.bias, 0.05)

    def forward(self, tfs):
        l1 = self.relu(self.l1(tfs))
        l2 = self.relu(self.l2(l1))
        l3 = self.relu(self.l3(l2))
        v = self.v(l3)
        return v

