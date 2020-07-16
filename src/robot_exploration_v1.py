import gym
import yaml
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sim_utils import draw_maps
from robot import Robot


class RobotExplorationT1(gym.Env):
    def __init__(self,config_path=os.getcwd()+'/../assets/config.yaml', number=None):
        np.random.seed(1234)
        with open(config_path) as stream:
            self.config = yaml.load(stream, Loader=yaml.SafeLoader)
        self.map_id_set_train = np.loadtxt(os.getcwd()+self.config['map_id_train_set'], str)
        self.map_id_set_eval = np.loadtxt(os.getcwd()+self.config['map_id_eval_set'],str)
        draw_maps(self.map_id_set_train,os.getcwd()+self.config['json_dir'],os.getcwd()+self.config['png_dir'])
        draw_maps(self.map_id_set_eval, os.getcwd() + self.config['json_dir'], os.getcwd() + self.config['png_dir'])
        if number is None:
            self.number = self.config['robots']['number']
        else:
            self.number = number
        # parameters will be set in reset func
        self.map_id = None
        self.frontiers = []
        self.target_points = []
        self.maze = np.zeros([1, 1])
        self.slam_map = np.zeros_like(self.maze)
        self.last_map = np.copy(self.slam_map)
        self.track_map = np.zeros_like(self.slam_map)
        self.data_transmitted = 0
        self.robots = []
        self.reset()

    def reset(self,random=True):
        if random:
            self.map_id = np.random.choice(self.map_id_set_train)
        else:
            self.map_id = self.map_id_set_eval[0]
            self.map_id_set_eval = np.delete(self.map_id_set_eval,0)
        print('map id： ',self.map_id)
        self.frontiers = [[] for i in range(self.config['frontiers']['number'])]
        self.target_points = []
        self.maze = self.map_loader(self.map_id)
        self.slam_map = np.ones_like(self.maze) * self.config['color']['uncertain']
        self.last_map = np.copy(self.slam_map)
        self.track_map = np.copy(self.maze)
        self.data_transmitted = 0
        self.robots = [Robot(i, np.copy(self.maze)) for i in range(self.number)]
        for rbt in self.robots:
            rbt.robot_list=self.robots
            rbt.world = self
            rbt.reset(np.copy(self.maze))
        self._merge_map()
        obs_n = []
        pose_n = []
        for i,rbt in enumerate(self.robots):
            obs_n.append(cv2.resize(rbt.get_obs(),(100,100),interpolation=cv2.INTER_NEAREST))
            pose = np.ones((1, self.number * 2)) * (-1)
            pose[:,2*i] = rbt.pose[0]
            pose[:,2*i+1] = rbt.pose[1]
            pose_n.append(pose)
        # if robots are in communication range, they communicate with others according to the mode
        if self.config['comm_mode'] == 'NC':
            # no communication
            pass
        else:
            for i, self_rbt in enumerate(self.robots):
                for j, other_rbt in enumerate(self.robots):
                    if not i==j:
                        if self.config['comm_mode'] == 'GC':
                            # complete communication
                            if np.linalg.norm(np.array(self_rbt.pose) - np.array(other_rbt.pose)) < self.config['robots']['commRange']:
                                self._communicate(self_rbt, other_rbt)
                        else:
                            # layers communication
                            if np.linalg.norm(np.array(self_rbt.pose) - np.array(other_rbt.pose)) < self.config['robots']['commRange']:
                                # exchange position information
                                pose_n[i][:,2*j] = other_rbt.pose[0]
                                pose_n[i][:,2*j+1] = other_rbt.pose[1]

                            if np.linalg.norm(np.array(self_rbt.pose)-np.array(other_rbt.pose)) < self.config['robots']['syncRange']:
                                # exchange complete information
                                self._communicate(self_rbt, other_rbt)
        return obs_n,pose_n

    def map_loader(self,map_id,padding=5):
        png_dir = os.getcwd()+self.config['png_dir']
        file_path = os.path.join(png_dir,map_id)+'.png'
        map_raw = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
        maze=np.zeros_like(map_raw)
        maze[map_raw==0] = self.config['color']['obstacle']
        maze[map_raw==255] = self.config['color']['free']
        index = np.where(maze==self.config['color']['obstacle'])
        [index_row_max,index_row_min,index_col_max,index_col_min] = [np.max(index[0]),np.min(index[0]),np.max(index[1]),np.min(index[1])]
        maze = maze[index_row_min:index_row_max+1,index_col_min:index_col_max+1]
        maze = np.lib.pad(maze, padding, mode='constant', constant_values=self.config['color']['obstacle'])
        maze = cv2.resize(maze,(self.config['map']['x'],self.config['map']['y']),interpolation=cv2.INTER_NEAREST)
        # map = cv2.dilate(map, np.ones((3, 3)), iterations=2)
        return maze

    def trackmap_loader(self,map_id,padding=5):
        png_dir = os.getcwd()+self.config['png_dir']
        file_path = os.path.join(png_dir, map_id) + '.png'
        map_raw = cv2.imread(file_path)
        map = cv2.resize(map_raw, (self.config['map']['x'], self.config['map']['y']), interpolation=cv2.INTER_NEAREST)
        return map

    def seed(self, seed=None):
        pass

    def _track(self):
        for i,rbt in enumerate(self.robots):
            for p in rbt.path:
                color = np.zeros(3)
                if i%2 == 0 :
                    color[0] = 255
                    cv2.circle(self.track_map, (p[1], p[0]), 2, color, -1)
                else:
                    color[2] = 255
                    cv2.rectangle(self.track_map, (p[1], p[0]), (p[1]+1, p[0]+1), color, -1)

    def render(self, mode='human'):
        state = np.copy(self._get_state())
        for rbt in self.robots:
            cv2.circle(state, (rbt.pose[1], rbt.pose[0]),rbt.robot_radius,color=self.config['color']['self'], thickness=-1)
        plt.figure(100)
        plt.clf()
        plt.imshow(state,cmap='gray')
        plt.pause(0.0001)

    def step(self, action_n):
        # action_n: 0~7
        obs_n = []
        rwd_n = []
        info_n = []
        pose_n = []
        for i, rbt in enumerate(self.robots):
            if action_n[i] == -1:
                # NOOP
                obs = rbt.get_obs()
                rwd = -2
                info = 'NOOP'
            else:
                obs, rwd, done, info = rbt.step(action_n[i])
            obs_n.append(cv2.resize(obs,(100,100),interpolation=cv2.INTER_NEAREST))
            rwd_n.append(rwd)
            info_n.append(info)
            pose = np.ones((1, self.number * 2)) * (-1)
            pose[:, 2 * i] = rbt.pose[0]
            pose[:, 2 * i + 1] = rbt.pose[1]
            pose_n.append(pose)
        self._merge_map()
        if self.config['comm_mode'] == 'NC':
            # no communication
            pass
        else:
            for i, self_rbt in enumerate(self.robots):
                for j, other_rbt in enumerate(self.robots):
                    if not i == j:
                        if self.config['comm_mode'] == 'LC':
                            if np.linalg.norm(np.array(self_rbt.pose) - np.array(other_rbt.pose)) < self.config['robots']['commRange']:
                                # layers communication
                                pose_n[i][:, 2 * j] = other_rbt.pose[0]
                                pose_n[i][:, 2 * j + 1] = other_rbt.pose[1]
                                self.data_transmitted = self.data_transmitted+pose_n[i].size
                            if np.linalg.norm(np.array(self_rbt.pose) - np.array(other_rbt.pose)) < self.config['robots']['syncRange']:
                                # complete communication
                                self._communicate(self_rbt, other_rbt)
                                self.data_transmitted += self_rbt.slam_map.size
                        else:
                            if np.linalg.norm(np.array(self_rbt.pose) - np.array(other_rbt.pose)) < self.config['robots']['commRange']:
                                # complete communication
                                self._communicate(self_rbt, other_rbt)
                                self.data_transmitted += self_rbt.slam_map.size

        # self.render()
        self._track()
        done = np.sum(self.slam_map == self.config['color']['free']) / np.sum(self.maze == self.config['color']['free']) > 0.95
        #if done:
            #self.track()
        return obs_n,rwd_n,done,info_n,pose_n

    def move_to_targets(self):
        """
        does the similar work as step func, but it dose not
        return anything, this func is used in GRE and RDM policy
        """
        for i,r in enumerate(self.robots):
            target_point = self.target_points[i]
            r.move_to_target(target_point)
        self._merge_map()
        for i, self_rbt in enumerate(self.robots):
            for j, other_rbt in enumerate(self.robots):
                if not i == j:
                    if np.linalg.norm(np.array(self_rbt.pose) - np.array(other_rbt.pose)) < self.config['robots']['syncRange']:
                        # complete communication
                        self._communicate(self_rbt, other_rbt)
        done = np.sum(self.slam_map == self.config['color']['free']) / np.sum(self.maze == self.config['color']['free']) > 0.95
        # self.render()
        self._track()
        return done

    def _communicate(self, rbt0, rbt1):
        bit_map = np.zeros_like(self.slam_map)
        merge_map = np.ones_like(self.slam_map) * 50
        for rbt in [rbt0,rbt1]:
            bit_map = np.bitwise_or(bit_map, rbt.slam_map != self.config['color']['uncertain'])
        idx = np.where(bit_map == 1)
        merge_map[idx] = self.maze[idx]
        rbt0.slam_map = merge_map
        rbt1.slam_map = merge_map
        return

    def get_frontiers(self):
        for rbt in self.robots:
            rbt.get_frontiers()

    def select_action_greedy(self):
        self.target_points=[]
        for r in self.robots:
            self.target_points.append(r.select_action_greedy())
        return self.target_points

    def select_action_randomly(self):
        act = []
        for rbt in self.robots:
            act.append(rbt.select_action_randomly())
        return act

    def select_target_randomly(self):
        self.target_points = []
        for rbt in self.robots:
            self.target_points.append(rbt.select_target_randomly())
        return None

    def close(self):
        pass

    def _merge_map(self):
        bit_map = np.zeros_like(self.slam_map)
        for rbt in self.robots:
            bit_map=np.bitwise_or(bit_map,rbt.slam_map!=self.config['color']['uncertain'])
        idx = np.where(bit_map==1)
        self.slam_map[idx]=self.maze[idx]
        return

    def _get_state(self):
        state = np.copy(self.slam_map)
        return state

    @property
    def path_length(self):
        length = []
        for _, rbt in enumerate(self.robots):
            length.append(rbt.path_length)
        return length


if __name__ == '__main__':
    env = RobotExplorationT1()
    for i in range(1000):
        obs_n = env.reset()
        for _ in range(100):
            action_n = np.random.randint(0,8,2)
            obs_n,rwd_n,done_n,info_n = env.step(action_n)
            print('reward:',np.sum(rwd_n))
        print('完成一个Episode，执行结果为',np.all(done_n))