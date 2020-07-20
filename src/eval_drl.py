import os

from maddpg.MADDPG import MADDPG
import torch
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
dim_states = 8
dim_actions = 8
MAX_STEPS = 100
TIMES = 10

CONFIG_PATH = os.getcwd()+'/../assets/config.yaml'
with open(CONFIG_PATH,'r') as stream:
    configs = yaml.load(stream, Loader=yaml.SafeLoader)
    n_agents = configs['robots']['number']


FloatTensor = torch.FloatTensor
print('Evaluate DRL!')
for times in range(TIMES):
    world = robot_exploration_v1.RobotExplorationT1()
    np.random.seed(times)
    torch.manual_seed(times)
    header = ['map_id','steps']
    data = []
    maddpg = MADDPG(n_agents, dim_states, dim_actions, 0, 0, -1)
    trackPath = os.getcwd() + '/../track/DRL/%s/'%times
    if not os.path.exists(trackPath):
        os.makedirs(trackPath)
    for i_episode in range(20):
        try:
            obs = world.reset(random=False)
        except Exception as e:
            print(e)
            continue
        obs = np.stack(obs)
        # history initialization
        length = 0
        for t in range(MAX_STEPS):
            action_probs = maddpg.select_action(obs).data.cpu()
            action_probs_valid = np.copy(action_probs)
            action = []
            for i, probs in enumerate(action_probs):
                rbt = world.robots[i]
                for j, frt in enumerate(rbt.get_frontiers()):
                    if len(frt) == 0:
                        action_probs_valid[i][j] = 0
                action.append(categorical.Categorical(probs=torch.tensor(action_probs_valid[i])).sample())
            action = torch.tensor(onehot_from_action(action))
            acts = np.argmax(action, axis=1)

            obs_, reward, done, _ = world.step(acts)
            length = length+np.sum(world.path_length)
            obs_ = np.stack(obs_)

            if not t == MAX_STEPS - 1:
                pass
            elif done:
                obs_ = None
            else:
                obs_ = None
            obs=obs_
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