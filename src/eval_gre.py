import os

from robot_exploration_v1 import RobotExplorationT1
import numpy as np
import torch as t
import time
import csv
import cv2
import yaml


n_agents = 2
action_dim = 8
MAX_STEPS = 100
TIMES = 10

CONFIG_PATH = os.getcwd()+'/../assets/config.yaml'
with open(CONFIG_PATH,'r') as stream:
    configs = yaml.load(stream, Loader=yaml.SafeLoader)
    n_agents = configs['robots']['number']

print('Evaluate GRE')
for times in range(TIMES):
    world = RobotExplorationT1()
    np.random.seed(times)
    t.manual_seed(times)
    header = ['map_id','steps']
    data = []
    trackPath = os.getcwd()+'/../track/GRE/%s/'%(times)
    if not os.path.exists(trackPath):
        os.makedirs(trackPath)
    for episode_i in range(20):
        obs =  world.reset(random=False)
        length = 0
        for step in range(MAX_STEPS):
            action = world.select_action_greedy()
            done = world.move_to_targets()
            length = length + np.sum(world.path_length)
            if done:
                print("Length: %d"%(length))
                cv2.imwrite(trackPath+world.map_id+'.png', world.track_map, params=None)
                data.append([world.map_id,length])
                break

    resultPath = os.getcwd()+'/../results/gre/'
    print(resultPath)
    os.makedirs(resultPath) if not os.path.exists(resultPath) else print('result path already exists')
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    fname = now + "_GRE_%d.csv"%(times)
    fpath = resultPath + fname
    with open(fpath, 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        f_csv.writerows(data)
    print('GRE Exploration Complete!')