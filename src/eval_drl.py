import os

from maddpg.MADDPG import MADDPG
import torch as th
import robot_exploration_v1
import csv
import time
from copy import copy
import numpy as np
from torch.distributions import categorical
from sim_utils import onehot_from_action
import cv2
import yaml

n_agents = 2
n_states = 2
n_actions = 8
dim_pose = 2
MAX_STEPS = 100
TIMES = 10

CONFIG_PATH = os.getcwd()+'/../assets/config.yaml'
with open(CONFIG_PATH,'r') as stream:
    configs = yaml.load(stream, Loader=yaml.SafeLoader)
    n_agents = configs['robots']['number']


FloatTensor = th.FloatTensor
print('Evaluate DRL!')
for times in range(TIMES):
    world = robot_exploration_v1.RobotExplorationT1()
    np.random.seed(times)
    th.manual_seed(times)
    header = ['map_id','steps']
    data = []
    maddpg = MADDPG(n_agents, n_states, n_actions, dim_pose, 0, 0, -1)
    trackPath = os.getcwd() + '/../track/DRL/%s/'%times
    if not os.path.exists(trackPath):
        os.makedirs(trackPath)
    for i_episode in range(20):
        try:
            obs,pose = world.reset(random=False)
            pose = th.tensor(pose)
        except Exception as e:
            print(e)
            continue
        obs = np.stack(obs)
        # history initialization
        obs_t_minus_0 = copy(obs)
        obs_t_minus_1 = copy(obs)
        obs_t_minus_2 = copy(obs)
        obs_t_minus_3 = copy(obs)
        obs_t_minus_4 = copy(obs)
        obs_t_minus_5 = copy(obs)
        obs = th.from_numpy(obs)
        obs_history = np.zeros((n_agents, obs.shape[1] * 6, obs.shape[2]))
        for i in range(n_agents):
            obs_history[i] = np.vstack((obs_t_minus_0[i], obs_t_minus_1[i], obs_t_minus_2[i],
                                        obs_t_minus_3[i], obs_t_minus_4[i], obs_t_minus_5[i]))
        if isinstance(obs_history, np.ndarray):
            obs_history = th.from_numpy(obs_history).float()
        length = 0
        for t in range(MAX_STEPS):
            obs_history = obs_history.type(FloatTensor)
            action_probs = maddpg.select_action(obs_history, pose).data.cpu()
            action_probs_valid = np.copy(action_probs)
            action = []
            for i, probs in enumerate(action_probs):
                rbt = world.robots[i]
                for j, frt in enumerate(rbt.get_frontiers()):
                    if len(frt) == 0:
                        action_probs_valid[i][j] = 0
                action.append(categorical.Categorical(probs=th.tensor(action_probs_valid[i])).sample())
            action = th.tensor(onehot_from_action(action))
            acts = np.argmax(action, axis=1)

            obs_, reward, done, _, next_pose = world.step(acts)
            length = length+np.sum(world.path_length)
            next_pose = th.tensor(next_pose)
            reward = th.FloatTensor(reward).type(FloatTensor)
            obs_ = np.stack(obs_)
            obs_ = th.from_numpy(obs_).float()

            obs_t_minus_5 = copy(obs_t_minus_4)
            obs_t_minus_4 = copy(obs_t_minus_3)
            obs_t_minus_3 = copy(obs_t_minus_2)
            obs_t_minus_2 = copy(obs_t_minus_1)
            obs_t_minus_1 = copy(obs_t_minus_0)
            obs_t_minus_0 = copy(obs_)
            obs_history_ = np.zeros((n_agents, obs.shape[1] * 6, obs.shape[2]))
            for i in range(n_agents):
                obs_history_[i] = np.vstack((obs_t_minus_0[i], obs_t_minus_1[i], obs_t_minus_2[i],
                                             obs_t_minus_3[i], obs_t_minus_4[i], obs_t_minus_5[i]))
            if not t == MAX_STEPS - 1:
                next_obs_history = th.tensor(obs_history_)
            elif done:
                next_obs_history = None
            else:
                next_obs_history = None
            obs_history=next_obs_history
            pose = next_pose
            if done:
                print('Length: %d'%length)
                data.append([world.map_id, length])
                cv2.imwrite(trackPath+world.map_id+'_track.png',world.track_map)
                break

    resultPath = os.getcwd()+'/../results/DRL/'
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    fname = now + "_DRL_%d.csv" % (times)
    fpath = resultPath + fname
    with open(fpath,'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        f_csv.writerows(data)
    print('DRL Exploration Complete!')