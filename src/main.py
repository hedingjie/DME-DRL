# from madrl_environments.pursuit import MAWaterWorld_mod
import robot_exploration_v1
from maddpg.MADDPG import MADDPG
import numpy as np
import torch
from tensorboardX import SummaryWriter
from copy import copy,deepcopy
from torch.distributions import categorical
from sim_utils import onehot_from_action
import time
import os
import yaml

# do not render the scene
e_render = False
# tensorboard writer
time_now = time.strftime("%m%d_%H%M%S")
writer = SummaryWriter(os.getcwd()+'/../runs/'+time_now)

food_reward = 10.
poison_reward = -1.
encounter_reward = 0.01
n_coop = 2
world = robot_exploration_v1.RobotExplorationT1()
reward_record = []

np.random.seed(1234)
torch.manual_seed(1234)
world.seed(1234)
n_agents = world.number
n_states = 8
n_actions = 8
n_pose = 2
# capacity = 1000000
capacity = 5000
# batch_size = 1000
batch_size = 100

n_episode = 200000
# max_steps = 1000
max_steps = 50
# episodes_before_train = 1000
episodes_before_train = 10

win = None
param = None
avg = None
load_model = False
CONFIG_PATH = os.getcwd() + '/../assets/config.yaml'
MODEL_DIR = os.getcwd() + '/../model/'

maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity,
                episodes_before_train)
with open(CONFIG_PATH,'r') as stream:
    config = yaml.safe_load(stream)

if load_model:
    if not os.path.exists(MODEL_DIR):
        pass
    else:
        checkpoints = torch.load(MODEL_DIR + '/model/model-%d.pth' % (config['robots']['number']))
        for i, actor in enumerate(maddpg.actors):
            actor.load_state_dict(checkpoints['actor_%d' % (i)])
            maddpg.actors_target[i] = deepcopy(actor)
        for i, critic in enumerate(maddpg.critics):
            critic.load_state_dict(checkpoints['critic_%d' % (i)])
            maddpg.critics_target[i] = deepcopy(critic)
        for i, actor_optim in enumerate(maddpg.actor_optimizer):
            actor_optim.load_state_dict(checkpoints['actor_optim_%d' % (i)])
        for i, critic_optim in enumerate(maddpg.critic_optimizer):
            critic_optim.load_state_dict(checkpoints['critic_optim_%d' % (i)])


FloatTensor = torch.cuda.FloatTensor if maddpg.use_cuda else torch.FloatTensor
for i_episode in range(n_episode):
    try:
        obs = world.reset()
    except Exception as e:
        continue
    obs = np.stack(obs)
    total_reward = 0.0
    rr = torch.zeros((n_agents), requires_grad=False)
    for t in range(max_steps):
        action_probs = maddpg.select_action(obs).data.cpu()
        action_probs_valid = np.copy(action_probs)
        action = []
        for i,probs in enumerate(action_probs):
            rbt = world.robots[i]
            for j,frt in enumerate(rbt.get_frontiers()):
                if len(frt) == 0:
                    action_probs_valid[i][j] = 0
            action.append(categorical.Categorical(probs=torch.tensor(action_probs_valid[i])).sample())

        action = onehot_from_action(action)
        acts = np.argmax(action,axis=1)
        for i in range(len(acts)):
            if len(world.robots[i].frontiers[acts[i]]) == 0:
                # NOOP 指令
                acts[i] = -1

        obs_, reward, done, _ = world.step(acts)

        if done:
            obs_ = None
        total_reward += np.sum(reward)
        rr += torch.tensor(reward, requires_grad=False)

        maddpg.memory.push(obs, action, obs_, reward, done)
        obs = obs_
        if t % 10 == 0:
            c_loss, a_loss = maddpg.update_policy()
        if done:
            break

    # if not discard:
    maddpg.episode_done += 1
    if maddpg.episode_done % 100 == 0:
        print('Save Models......')
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        dicts = {}
        for i in range(maddpg.n_agents):
            dicts['actor_%d' % (i)] = maddpg.actors_target[i].state_dict()
            dicts['critic_%d' % (i)] = maddpg.critics_target[i].state_dict()
            dicts['actor_optim_%d' % (i)] = maddpg.actor_optimizer[i].state_dict()
            dicts['critic_optim_%d' % (i)] = maddpg.critic_optimizer[i].state_dict()
        torch.save(dicts, MODEL_DIR + '/model-%d.pth' % (config['robots']['number']))
    print('Episode: %d, reward = %f' % (i_episode, total_reward))
    reward_record.append(total_reward)
    # visual
    writer.add_scalars('scalar/reward',{'total_rwd':total_reward,'r0_rwd':rr[0],'r1_rwd':rr[1]},i_episode)
    if i_episode > episodes_before_train and i_episode % 10 == 0:
        writer.add_scalars('scalar/mean_rwd',{'mean_reward':np.mean(reward_record[-100:])}, i_episode)
    if not c_loss is None:
        writer.add_scalars('loss/c_loss',{'r0':c_loss[0],'r1':c_loss[1]},i_episode)
    if not a_loss is None:
        writer.add_scalars('loss/a_loss',{'r0':a_loss[0],'r1':a_loss[1]},i_episode)

    if maddpg.episode_done == maddpg.episodes_before_train:
        print('training now begins...')

world.close()