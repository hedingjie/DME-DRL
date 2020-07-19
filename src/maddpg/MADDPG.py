from maddpg.model import Critic, Actor
import torch
from copy import deepcopy
from maddpg.memory import ReplayMemory, Experience
from torch.optim import Adam
import torch.nn as nn
import numpy as np
from sim_utils import gumbel_softmax

LOAD_MODEL = False

def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)


class MADDPG:
    def __init__(self, n_agents, dim_obs, dim_act, batch_size,
                 capacity, episodes_before_train):

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        # self.use_cuda = t.cuda.is_available()
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu:0')
        self.episodes_before_train = episodes_before_train

        self.GAMMA = 0.95
        self.tau = 0.01

        # self.var = [1.0 for i in range(n_agents)]
        self.var = [0.01 for i in range(n_agents)]

        self.actors = [Actor(dim_obs, dim_act).to(self.device) for i in range(n_agents)]
        self.critics = [Critic(n_agents, dim_obs, dim_act).to(self.device) for i in range(n_agents)]
        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=0.001) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=0.001) for x in self.actors]

        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        # if self.use_cuda:
        #     for x in self.actors:
        #         x.cuda()
        #     for x in self.critics:
        #         x.cuda()
        #     for x in self.actors_target:
        #         x.cuda()
        #     for x in self.critics_target:
        #         x.cuda()

        self.steps_done = 0
        self.episode_done = 0

    def update_policy(self):
        # 直到收集到足够的经验才开始训练
        if self.episodes_before_train == -1:
            return None,None
        if self.episode_done <= self.episodes_before_train:
            return None,None

        ByteTensor = torch.cuda.ByteTensor if self.use_cuda else torch.ByteTensor
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        c_loss = []
        a_loss = []
        for agent in range(self.n_agents):
            trainsitions = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*trainsitions))
            non_final_mask = ByteTensor(list(map(lambda s:s is not None, batch.next_states))).bool()
            state_batch = torch.from_numpy(np.stack(batch.states,axis=0)).type(FloatTensor)
            action_batch = torch.from_numpy(np.stack(batch.actions,axis=0)).type(FloatTensor)
            reward_batch = torch.from_numpy(np.stack(batch.rewards,axis=0)).type(FloatTensor)
            non_final_next_states = torch.from_numpy(
                np.stack([s for s in batch.next_states if s is not None],axis=0)
            ).type(FloatTensor)

            whole_state = state_batch
            whole_action = action_batch
            self.critic_optimizer[agent].zero_grad()
            current_Q = self.critics[agent](whole_state,whole_action)

            non_final_next_actions = [
                self.actors_target[i](non_final_next_states[:,i,:]) for i in range(self.n_agents)
            ]
            non_final_next_actions = torch.stack(non_final_next_actions)
            non_final_next_actions = (non_final_next_actions.transpose(0,1).contiguous())

            target_Q = torch.zeros(self.batch_size).type(FloatTensor)
            target_Q[non_final_mask] = self.critics_target[agent](
                non_final_next_states,
                non_final_next_actions,
            ).squeeze()
            target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (reward_batch[:, agent].unsqueeze(1))

            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            loss_Q.backward()
            self.critic_optimizer[agent].step()

            self.actor_optimizer[agent].zero_grad()
            state_i = state_batch[:, agent, :]
            action_i = self.actors[agent](state_i)
            ac = action_batch.clone()
            ac[:, agent, :] = action_i
            whole_action = ac.view(self.batch_size, self.n_agents ,-1)
            actor_loss = -self.critics[agent](whole_state, whole_action)
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            self.actor_optimizer[agent].step()

            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        if self.steps_done % 100 == 0 and self.steps_done > 0:
            print('Softupdate')
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

        return c_loss, a_loss

    def select_action(self, state_batch):
        # state_batch: n_agents x state_dim
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        with torch.no_grad():
            actions = torch.zeros(self.n_agents,self.n_actions).type(FloatTensor)
            state_batch = torch.tensor(state_batch,dtype=torch.float).type(FloatTensor)
            for i in range(self.n_agents):
                state = state_batch[i, :]
                act = self.actors[i](state).data.cpu()
                act_prob=gumbel_softmax(act).squeeze()
                act_prob = torch.clamp(act_prob, 1e-6, 1 - 1e-6)
                actions[i, :] = act_prob
            self.steps_done += 1
            return torch.squeeze(actions)