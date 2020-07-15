import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sim_utils import gumbel_softmax

from maddpg import basic_module

class Critic(basic_module.BasicModule):
    def __init__(self, n_agent, dim_observation, dim_action, dim_pose):
        super(Critic, self).__init__()

        # RMADDPG
        self.hidden_dim = 256
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        self.conv1 = nn.Conv2d(self.n_agent,16,8,4)
        self.conv2 = nn.Conv2d(16,32,4,2)
        self.i2h1 = nn.Linear(4128,self.hidden_dim)
        self.rnn = nn.LSTM(3872, self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim+dim_action*n_agent+dim_pose*n_agent*n_agent,1)


    # obs: batch_size * obs_dim
    def forward(self, obs, acts, poses):
        # obs' shape: batch_size x agent_number x observation's shape
        b, n, r, c = obs.shape
        obs_5 = obs[:,:, 0:1 * int(r / 6)]
        obs_4 = obs[:,:, 1 * int(r / 6):2 * int(r / 6)]
        obs_3 = obs[:,:, 2 * int(r / 6):3 * int(r / 6)]
        obs_2 = obs[:,:, 3 * int(r / 6):4 * int(r / 6)]
        obs_1 = obs[:,:, 4 * int(r / 6):5 * int(r / 6)]
        obs_0 = obs[:,:, 5 * int(r / 6):]

        hist_0 = F.relu(self.conv2(F.relu(self.conv1(obs_0))))
        hist_0 = hist_0.contiguous().view(-1, self.num_flat_features(hist_0))
        hist_1 = F.relu(self.conv2(F.relu(self.conv1(obs_1))))
        hist_1 = hist_1.contiguous().view(-1, self.num_flat_features(hist_1))
        hist_2 = F.relu(self.conv2(F.relu(self.conv1(obs_2))))
        hist_2 = hist_2.contiguous().view(-1, self.num_flat_features(hist_2))
        hist_3 = F.relu(self.conv2(F.relu(self.conv1(obs_3))))
        hist_3 = hist_3.contiguous().view(-1, self.num_flat_features(hist_3))
        hist_4 = F.relu(self.conv2(F.relu(self.conv1(obs_4))))
        hist_4 = hist_4.contiguous().view(-1, self.num_flat_features(hist_4))
        hist_5 = F.relu(self.conv2(F.relu(self.conv1(obs_5))))
        hist_5 = hist_5.contiguous().view(-1, self.num_flat_features(hist_5))


        hist_obs = t.stack((hist_0, hist_1, hist_2, hist_3, hist_4, hist_5))
        # hist_obs = t.stack((hist_0, hist_1, hist_2))
        batch_size = hist_obs.shape[1]
        h0 = t.randn(1, batch_size, 256)
        c0 = t.randn(1, batch_size, 256)
        _, (hn, cn) = self.rnn(hist_obs, (h0, c0))
        s, b, h = hn.shape
        out = hn.contiguous().view(b,-1)
        b,n,d = acts.shape
        acts_ = acts.contiguous().view(b,-1)
        b,n,_,d = poses.shape
        poses = poses.contiguous().view(b,-1)
        out = t.cat((out,acts_,poses),dim=1)
        value = self.fc(out)
        return value

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



class Actor(basic_module.BasicModule):
    def __init__(self, n_agent, dim_pose):
        super(Actor, self).__init__()
        # RNN
        self.hidden_dim = 256
        out_dim = 8
        self.conv1 = nn.Conv2d(1,16,8,4)
        self.conv2 = nn.Conv2d(16,32,4,2)
        self.i2h1 = nn.Linear(4128,self.hidden_dim)
        self.rnn = nn.LSTM(3872,self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim+dim_pose*n_agent,8)
        # self.fc = nn.Linear(self.hidden_dim, 8)

    def forward(self, obs, poses):
        # NCHW 转 NHWC
        # obs' shape: batch_size x agent_number x observation's shape
        # obs = (obs-t.min(obs))/(t.max(obs)-t.min(obs))
        poses = poses.type(t.float32)
        _,r,c = obs.shape
        obs_5 = t.unsqueeze(obs[:,0:1*int(r/6)],dim=1)
        obs_4 = t.unsqueeze(obs[:,1*int(r/6):2*int(r/6)],dim=1)
        obs_3 = t.unsqueeze(obs[:,2*int(r/6):3*int(r/6)],dim=1)
        obs_2 = t.unsqueeze(obs[:,3*int(r/6):4*int(r/6)],dim=1)
        obs_1 = t.unsqueeze(obs[:,4*int(r/6):5*int(r/6)],dim=1)
        obs_0 = t.unsqueeze(obs[:,5*int(r/6):],dim=1)

        hist_0 = F.relu(self.conv2(F.relu(self.conv1(obs_0))))
        hist_0 = hist_0.contiguous().view(-1,self.num_flat_features(hist_0))
        hist_1 = F.relu(self.conv2(F.relu(self.conv1(obs_1))))
        hist_1 = hist_1.contiguous().view(-1, self.num_flat_features(hist_1))
        hist_2 = F.relu(self.conv2(F.relu(self.conv1(obs_2))))
        hist_2 = hist_2.contiguous().view(-1, self.num_flat_features(hist_2))
        hist_3 = F.relu(self.conv2(F.relu(self.conv1(obs_3))))
        hist_3 = hist_3.contiguous().view(-1, self.num_flat_features(hist_3))
        hist_4 = F.relu(self.conv2(F.relu(self.conv1(obs_4))))
        hist_4 = hist_4.contiguous().view(-1, self.num_flat_features(hist_4))
        hist_5 = F.relu(self.conv2(F.relu(self.conv1(obs_5))))
        hist_5 = hist_5.contiguous().view(-1, self.num_flat_features(hist_5))

        hist_obs = t.stack((hist_0,hist_1,hist_2,hist_3,hist_4,hist_5))
        batch_size = hist_obs.shape[1]
        h0 = t.randn(1,batch_size,256)
        c0 = t.randn(1,batch_size,256)
        _,(hn,cn) = self.rnn(hist_obs,(h0,c0))
        s,b,h = hn.shape
        out = hn.contiguous().view(b,-1)
        b, _, d = poses.shape
        poses = poses.contiguous().view(b, -1)
        out = t.cat((out,poses),dim=1)
        action = self.fc(out)
        action = gumbel_softmax(action)
        return action

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features