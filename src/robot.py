import numpy as np
import yaml
import cv2
import matplotlib.pyplot as plt
import sim_utils
from navigate import AStar, AStarSimple
import os


class Robot():

    def __init__(self, rbt_id, maze, config_path=os.getcwd() + '/../assets/config.yaml'):
        with open(config_path) as stream:
            self.config = yaml.load(stream, Loader=yaml.SafeLoader)
        self.id = rbt_id
        self.maze = maze
        self.robot_radius = self.config['robots']['robotRadius']
        self.comm_range = self.config['robots']['commRange']
        self.sync_range = self.config['robots']['syncRange']
        self.laser_range = self.config['laser']['range']
        self.laser_fov = self.config['laser']['fov']
        self.laser_resol = self.config['laser']['resolution']
        self.state_size = (self.config['stateSize']['y'], self.config['stateSize']['x'])
        self.slam_map = np.ones_like(self.maze) * self.config['color']['uncertain']
        self.pose = self._init_pose()
        self.last_map = self.slam_map.copy()
        self.navigator = AStar()
        self.robot_list = None
        self.world = None
        self.path = None
        self.frontiers = []

        """ pre calculate radius and angle vector that will be used in building map """
        radius_vect = np.arange(self.laser_range + 1)
        self._radius_vect = radius_vect.reshape(1, radius_vect.shape[0])
        # generate radius vector of [0,1,2,...,laser_range]

        angles_vect = np.arange(-self.laser_fov * 0.5, self.laser_fov * 0.5, step=self.laser_resol)
        self._angles_vect = angles_vect.reshape(angles_vect.shape[0], 1)
        # generate angles vector from -laser_angle/2 to laser_angle

    def _init_pose(self):
        if self.config['robots']['resetRandomPose'] == 1:
            h, w = self.maze.shape
            y_min, y_max = int(0.1 * h), int(0.8 * h)
            x_min, x_max = int(0.1 * w), int(0.8 * w)
            y = np.random.randint(y_min, y_max)
            x = np.random.randint(x_min, x_max)
            while self.maze[y, x] == self.config['color']['obstacle']:
                y = np.random.randint(y_min, y_max)
                x = np.random.randint(x_min, x_max)
            return y, x
        else:
            return self.config['robots']['startPose']['y'], self.config['robots']['startPose']['x']

    def reset(self, maze):
        self.maze = maze
        self.pose = self._init_pose()
        self.slam_map = np.ones_like(self.maze) * self.config['color']['uncertain']
        self.last_map = self.slam_map.copy()
        self._build_map()
        self.last_map = np.copy(self.slam_map)
        obs = self.get_obs()
        return obs

    def render(self):
        state = self.get_obs()
        plt.figure(self.id)
        plt.clf()
        plt.imshow(state, cmap='gray')
        plt.pause(0.0001)

    def _build_map_with_rangeCoordMat(self, y_range_coord_mat, x_range_coord_mat):
        """build map after moving"""
        y_range_coord_mat = (np.round(y_range_coord_mat)).astype(np.int)
        x_range_coord_mat = (np.round(x_range_coord_mat)).astype(np.int)
        in_bound_ind = sim_utils.within_bound(np.array([y_range_coord_mat, x_range_coord_mat]), self.maze.shape)

        """delete points outside boundaries"""
        outside_ind = np.argmax(~in_bound_ind, axis=1)  # 找到每一个角度上的第一个不在世界范围内的索引，为[极径]
        ok_ind = np.where(outside_ind == 0)[0]  # 当这个索引为0时表示这个角度的射线可以不用考虑（因为根本就不发出射线），为[极角]
        need_amend_ind = np.where(outside_ind != 0)[0]  # 真正需要修改的是那些发出了射线但是中途被截断的，need_amend_ind的数值表示的是度数，为[极角]
        outside_ind = np.delete(outside_ind, ok_ind)  # 从outside_ind去除不用考虑的射线，为[极径]
        inside_ind = np.copy(outside_ind)  # index_ind表示的是每一个角度中在世界范围内的射线，为[极径]
        inside_ind[inside_ind != 0] -= 1  # -1使得目标达到inside_ind的边界，为[极径]
        bound_ele_x = x_range_coord_mat[
            need_amend_ind, inside_ind]  # [need_amend_ind,inside_ind]表示的是在世界范围内的矩阵，为探测范围内的边缘点的x值
        bound_ele_y = y_range_coord_mat[need_amend_ind, inside_ind]  # [need_amend_ind,inside_ind]与上同理，为探测范围内的边缘点的y值

        count = 0
        for i in need_amend_ind:
            x_range_coord_mat[i, ~in_bound_ind[i, :]] = bound_ele_x[count]  # 将每一条射线不在世界范围内的坐标都设置为其边缘点的坐标，这样修改便不会延伸
            y_range_coord_mat[i, ~in_bound_ind[i, :]] = bound_ele_y[count]
            count += 1

        """find obstacles along the (laser) ray"""
        x_rangeCoordMat_ = np.clip(x_range_coord_mat, 0, self.maze.shape[1] - 1)
        y_rangeCoordMat_ = np.clip(y_range_coord_mat, 0, self.maze.shape[0] - 1)
        obstacle_ind = np.argmax(self.maze[y_rangeCoordMat_, x_rangeCoordMat_] ==
                                 self.config['color']['obstacle'], axis=1)  # 找到每一条射线中极径最小的点，存储为[极径]，即被遮挡
        obstacle_ind[obstacle_ind == 0] = x_range_coord_mat.shape[1]  # 对于紧挨着障碍物的点，将它的极径设置为最远，即未被遮挡

        """产生形如[[1,2,3,...],[1,2,3,...],[1,2,3,...]]的矩阵来与障碍物的坐标进行比较"""
        bx = np.arange(x_range_coord_mat.shape[1]).reshape(1, x_range_coord_mat.shape[1])
        by = np.ones((x_range_coord_mat.shape[0], 1))
        b = np.matmul(by, bx)

        """获取机器人可以感知的点的坐标"""
        b = b <= obstacle_ind.reshape(obstacle_ind.shape[0], 1)  # 未被遮挡的点的坐标存储在b中
        y_coord = y_range_coord_mat[b]
        x_coord = x_range_coord_mat[b]

        """ no slam error """
        self.slam_map[y_coord, x_coord] = self.maze[y_coord, x_coord]

        """ dilate/close to fill the holes """
        # self.dslamMap= cv2.morphologyEx(self.slamMap,cv2.MORPH_CLOSE,np.ones((3,3)))
        # self.slam_map = cv2.dilate(self.slam_map, np.ones((3, 3)), iterations=1)
        return self.slam_map

    def _build_map(self):
        pose = self.pose
        y_range_coord_mat = pose[0] - np.matmul(np.sin(self._angles_vect), self._radius_vect)
        x_range_coord_mat = pose[1] + np.matmul(np.cos(self._angles_vect), self._radius_vect)
        self._build_map_with_rangeCoordMat(y_range_coord_mat, x_range_coord_mat)
        return np.copy(self.slam_map)

    def _move_one_step(self, next_point):
        if not self._is_crashed(next_point):
            self.pose = next_point
            map_temp = np.copy(self.slam_map)  # 临时地图，存储原有的slam地图
            self._build_map()
            map_incrmnt = np.count_nonzero(map_temp - self.slam_map)  # map increment
            # self.render()
            return map_incrmnt
        else:
            return -1

    def step(self, action):
        y, x = self.pose
        y_dsti, x_dsti = self.frontiers[action][0]
        distance_min = np.sqrt((y - y_dsti) ** 2 + (x - x_dsti) ** 2)
        for (y_, x_) in self.frontiers[action]:
            distance = np.sqrt((y - y_) ** 2 + (x - x_) ** 2)
            if distance < distance_min:
                y_dsti, x_dsti = y_, x_
                distance_min = distance
        self.destination = (y_dsti, x_dsti)
        self.path = self.navigator.navigate(self.maze, self.pose, self.destination)
        counter = 0
        if self.path is None:
            raise Exception('The target point is not accessible')
        else:
            incrmnt_his = []  # map increament list, record the history of it
            for i, point in enumerate(self.path):
                counter += 1
                map_incrmnt = self._move_one_step(point)
                incrmnt_his.append(map_incrmnt)
                if np.sum(incrmnt_his) > 3600:
                    # print('地图增量超过阈值，提前终止探索过程')
                    break
        obs = self.get_obs()
        rwd = self.reward(counter, incrmnt_his)
        done = np.sum(self.slam_map == self.config['color']['free']) / np.sum(
            self.maze == self.config['color']['free']) > 0.95
        info = 'Robot %d has moved to the target point' % (self.id)
        return obs, rwd, done, info

    def move_to_target(self, target):
        """
        move robot to the target position
        :param target: target position, type of np.array
        :return:
        """
        self.destination = target
        if target is None:
            obs = self.get_obs()
            done = np.sum(self.slam_map == self.config['color']['free']) / np.sum(
                self.maze == self.config['color']['free']) > 0.95
            info = "No.%d robot fails to move." % self.id
            return obs, 0, done, info
        self.path = self.navigator.navigate(self.maze, self.pose, self.destination)
        if self.path is None:
            raise Exception("The target point is not accessible")
        for point in self.path:
            self._move_one_step(point)
        obs = self.get_obs()
        done = np.sum(self.slam_map == self.config['color']['free']) / np.sum(
            self.maze == self.config['color']['free']) > 0.95
        info = "No.%d robot moves successfully." % self.id
        return obs, None, done, info

    def reward(self, counter, incrmnt_his):
        """reward function"""
        rwd1 = np.sum(incrmnt_his) * self.config['robots']['w1']
        rwd2 = -1. * counter * self.config['robots']['w2']
        rwd = rwd1 + rwd2
        return rwd

    def get_state(self):
        state = self.slam_map.copy()
        for rbt in self.robot_list:
            if rbt.id == self.id:
                cv2.rectangle(state, (rbt.pose[1] - rbt.robot_radius, rbt.pose[0] - rbt.robot_radius),
                              (rbt.pose[1] + rbt.robot_radius, rbt.pose[0] + rbt.robot_radius),
                              color=self.config['color']['self'], thickness=-1)
            else:
                cv2.rectangle(state, (rbt.pose[1] - rbt.robot_radius, rbt.pose[0] - rbt.robot_radius),
                              (rbt.pose[1] + rbt.robot_radius, rbt.pose[0] + rbt.robot_radius),
                              color=self.config['color']['others'], thickness=-1)
        return state.copy()

    def _is_crashed(self, target_point):
        if not sim_utils.within_bound(target_point, self.maze.shape, 0):
            return True
        # 将机器人视为一个质点
        y, x = target_point
        return self.maze[y, x] == self.config['color']['obstacle']

    def get_obs(self):
        """每一个机器人都获取自己观察视野内的本地地图"""
        observation = self.get_state()
        self.get_frontiers()
        return observation

    def get_frontiers(self):
        """获取前沿，采用地图相减的算法"""
        slam_map = np.copy(self.slam_map).astype(np.uint8)
        self.frontiers = [[] for _ in range(self.config['frontiers']['number'])]
        contour = []
        map = np.copy(self.maze).astype(np.uint8)
        clr_fre = self.config['color']['free']
        clr_obs = self.config['color']['obstacle']
        clr_unc = self.config['color']['uncertain']

        slam_map_ = np.copy(slam_map)
        slam_map_[slam_map_ == clr_fre] = 25
        slam_map_[slam_map_ == clr_obs] = 0
        slam_map_[slam_map_ == clr_unc] = 0
        canny_0 = cv2.Canny(slam_map_, 10, 25)
        canny_0[map == clr_obs] = 0

        slam_map_ = np.copy(slam_map)
        slam_map_[slam_map_ == clr_fre] = clr_unc
        canny_1 = cv2.Canny(slam_map_, 50, 100)

        canny_0[canny_1 == 255] = 0

        coord = np.where(canny_0 == 255)
        ry, rx = self.pose[0], self.pose[1]

        tan = (coord[0] - ry) / (coord[1] - rx)
        for idx, (y, x) in enumerate(zip(coord[0], coord[1])):
            if self.maze[y, x] == self.config['color']['free']:
                if x > rx:
                    if 1 <= tan[idx]:
                        self.frontiers[0].append((y, x))
                    elif 0 <= tan[idx] < 1:
                        self.frontiers[1].append((y, x))
                    elif -1 <= tan[idx] < 0:
                        self.frontiers[2].append((y, x))
                    elif tan[idx] < -1:
                        self.frontiers[3].append((y, x))
                else:
                    if 1 <= tan[idx]:
                        self.frontiers[4].append((y, x))
                    elif 0 <= tan[idx] < 1:
                        self.frontiers[5].append((y, x))
                    elif -1 <= tan[idx] < 0:
                        self.frontiers[6].append((y, x))
                    elif tan[idx] < -1:
                        self.frontiers[7].append((y, x))

        if len(self.frontiers) > 0:
            return np.copy(self.frontiers)
        else:
            raise Exception('Exception: None Contour!')

    def eular_dis(self, point):
        try:
            x, y = point
            return np.linalg.norm([self.pose[0] - x, self.pose[1] - y])
        except TypeError:
            return np.inf

    def select_action(self):
        f_points = [point for front in self.frontiers for point in front]
        f_dis = list(map(self.eular_dis, f_points))
        return f_points[np.argmin(f_dis)]

    def center_point(self, x):
        return np.mean(x, axis=0)

    def select_action_greedy(self):
        centers = list(map(self.center_point, self.frontiers))
        cen_dis = list(map(self.eular_dis, centers))
        action = np.argmin(cen_dis)
        f_dis = list(map(self.eular_dis, self.frontiers[action]))
        return self.frontiers[action][np.argmin(f_dis)]
        # return self.select_action()

    def select_action_randomly(self):
        action = np.random.randint(8)
        while len(self.frontiers[action]) == 0:
            action = np.random.randint(8)
        return action

    def select_target_randomly(self):
        f_points = [point for front in self.frontiers for point in front]
        target = f_points[np.random.randint(0, len(f_points))]
        return target

    @property
    def path_length(self):
        return len(self.path)
