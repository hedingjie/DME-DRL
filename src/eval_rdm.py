import os

import torch as th
import numpy as np
import robot_exploration_v1
import csv
import time
import cv2
import yaml

MAX_STEPS = 100
TIMES = 10

CONFIG_PATH = os.getcwd()+'/../assets/config.yaml'
with open(CONFIG_PATH,'r') as stream:
    configs = yaml.load(stream, Loader=yaml.SafeLoader)
    n_agents = configs['robots']['number']

print('Evaluate RDM')
for times in range(TIMES):
    world = robot_exploration_v1.RobotExplorationT1()
    np.random.seed(times)
    th.manual_seed(times)
    header = ['map_id', 'steps']
    data = []
    trackPath = os.getcwd() + '/../track/RDM/%s/'%times
    if not os.path.exists(trackPath):
        os.makedirs(trackPath)
    for i_episode in range(20):
        try:
            world.reset(random=False)
        except Exception as e:
            print(e)
            continue
        if i_episode <= -1:
            continue
        length = 0
        for t in range(MAX_STEPS):
            try:
                world.get_frontiers()
            except Exception as e:
                print(e)
                break
            act_gre = world.select_target_randomly()
            try:
                done = world.move_to_targets()
                length = length + np.sum(world.path_length)
            except Exception as e:
                print(e)
                break
            if done:
                print(length)
                print(trackPath)
                cv2.imwrite(trackPath+world.map_id+'.png', world.track_map, params=None)
                data.append([world.map_id, length])
                break
    resultPath = os.getcwd() + '/results/RDM/'
    print(resultPath)
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    fname = now + "_RDM_%d.csv"%(times)
    fpath = resultPath + fname
    with open(fpath,'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        f_csv.writerows(data)
    print('RDM Exploration Complete!')