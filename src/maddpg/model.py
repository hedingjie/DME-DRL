import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sim_utils import gumbel_softmax

from maddpg import basic_module

class Critic(basic_module.BasicModule):
    def __init__(self, n_agent, dim_observation, dim_action):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        self.fc1 = nn.Sequential(
            nn.Linear(n_agent * (dim_observation+dim_action), 64),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(n_agent * dim_observation, 128),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(n_agent * dim_observation, 64),
            nn.ReLU()
        )
        self.fc4 = nn.Linear(64,1)


    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        # obs' shape: batch_size x agent_number x observation's shape
        x = torch.cat(obs, acts)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = nn.Softmax(x)
        return x



class Actor(basic_module.BasicModule):
    def __init__(self, dim_observation, dim_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim_observation,64),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64,128),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU()
        )
        self.fc4 = nn.Linear(64,dim_action)


    def forward(self, obs):
        x = self.fc1(obs)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        action = gumbel_softmax(x)
        return action