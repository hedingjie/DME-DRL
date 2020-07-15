import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import cv2


# def distance(node1, node2):
#     return max(abs(node1.i-node2.i),abs(node1.j-node2.j))

height = 200
width = 200

class Node(object):
    def __init__(self,father,i,j,end_i,end_j):
        if i<0 or i >= height or j<0 or j>= width:
            raise Exception("节点数组越界")
        else:
            self.father = father
            self.i=i
            self.j=j

        if father != None:
            if abs(self.i-father.i) == 1 and abs(self.j - father.j) == 1:
                self.G = father.G + 1.414
            else:
                self.G = father.G + 1
            x_dis = np.abs(self.i-end_i)
            y_dis = np.abs(self.j-end_j)
            self.H = x_dis + y_dis + (1.414 - 2) * min(x_dis, y_dis)
            self.F = self.G+self.H
        else:
            self.G = 0
            self.H = 0
            self.F = 0

    def reset_father(self,father,new_G):
        if father != None:
            self.G = new_G
            self.F = self.G+self.H

        self.father = father

    def distance(self,end_i,end_j):
        return max(abs(self.i - end_i), abs(self.j - end_j))


class AStar:
    def __init__(self,config_path=os.getcwd()+'/../assets/config.yaml'):
        self._map_color = {'uncertain': 50, 'free': 0, 'obstacle': 100, 'self': 250,'others':200}
        with open(config_path) as stream:
            self.config = yaml.load(stream, Loader=yaml.SafeLoader)
        self._open_list = {}
        self._close_list = {}
        self._path = None

    def get_path(self):
        return self._path

    def _min_F_node(self):
        if len(self._open_list) == 0:
            raise Exception('路径不存在！')

        _min = 9999999999999999
        _k= self._start

        for k,v in self._open_list.items():
            if _min > v.F:
                _min = v.F
                _k = k
        return self._open_list[_k]

    def _add_to_open(self,node):
        self._open_list.pop((node.i,node.j))
        self._close_list[(node.i,node.j)]=node
        adjacent = []

        # 添加相邻节点的时候要注意边界
        # 上
        try:
            adjacent.append(Node(node, node.i - 1, node.j,self._end.i,self._end.j))
        except Exception as err:
            pass
        # 下
        try:
            adjacent.append(Node(node, node.i + 1, node.j,self._end.i,self._end.j))
        except Exception as err:
            pass
        # 左
        try:
            adjacent.append(Node(node, node.i, node.j - 1,self._end.i,self._end.j))
        except Exception as err:
            pass
        # 右
        try:
            adjacent.append(Node(node, node.i, node.j + 1,self._end.i,self._end.j))
        except Exception as err:
            pass
        # 左上
        try:
            adjacent.append(Node(node, node.i - 1, node.j - 1,self._end.i,self._end.j))
        except Exception as err:
            pass
        # 左下
        try:
            adjacent.append(Node(node, node.i + 1, node.j - 1,self._end.i,self._end.j))
        except Exception as err:
            pass
        # 右上
        try:
            adjacent.append(Node(node, node.i - 1, node.j + 1,self._end.i,self._end.j))
        except Exception as err:
            pass
        # 右下
        try:
            adjacent.append(Node(node, node.i + 1, node.j + 1,self._end.i,self._end.j))
        except Exception as err:
            pass

        # 检查每一个相邻的点
        for a in adjacent:
            # 如果是终点，结束
            if (a.i, a.j) == (self._end.i, self._end.j):
                if abs(a.i-node.i) == 1 and abs(a.j-node.j) == 1:
                    new_G = node.G + 1.414
                    self._end.reset_father(node, new_G)
                else :
                    new_G = node.G + 1
                    self._end.reset_father(node,new_G)
                return True
            # 如果在close_list中,不去理他
            if (a.i, a.j) in self._close_list:
                continue
            # 如果不在open_list中，则添加进去
            if (a.i, a.j) not in self._open_list:
                self._open_list[(a.i, a.j)] = a
            # 如果存在在open_list中，通过G值判断这个点是否更近
            else:
                exist_node = self._open_list[(a.i, a.j)]
                if abs(exist_node.i-node.i) == 1 and abs(exist_node.j-node.j) == 1 :
                    new_G = node.G + 1.414
                else:
                    new_G = node.G + 1
                if new_G < exist_node.G:
                    exist_node.reset_father(node, new_G)

        return False

    # 查找路线
    def _find_the_path(self):
        self._open_list[(self._start.i, self._start.j)] = self._start

        the_node = self._start
        try:
            while not self._add_to_open(the_node):
                the_node = self._min_F_node()

        except Exception as err:
            # 路径找不到
            print(err)
            return False
        return True

    # 通过递归的方式根据每个点的父节点将路径连起来
    def _mark_path(self):
        node = self._end
        path_points = []
        while node.father != None:
            path_points.insert(0, (node.i, node.j))
            node = node.father
        return path_points

    def _preset_maze(self):
        (h,w) = self._maze.shape
        for i, row in enumerate(self._maze):
            for j,ele in enumerate(row):
                if ele==self.config['color']['obstacle']:
                    block_node=Node(None,i,j,self._end.i,self._end.j)
                    self._close_list[(block_node.i,block_node.j)]=block_node

    def navigate(self,maze,start_pose,end_pose):
        global height, width
        start_i, start_j = start_pose
        end_i, end_j = end_pose
        # reset
        self._open_list = {}
        self._close_list = {}
        self._path = None

        self._maze = maze
        height = maze.shape[0]
        width = maze.shape[1]
        self._start = Node(None, start_i, start_j, end_i, end_j)
        self._end = Node(None, end_i, end_j, end_i, end_j)
        self._preset_maze()
        if (self._start.i,self._start.j) in self._close_list:
            return None
        if (self._end.i,self._end.j) in self._close_list:
            return None
        if self._find_the_path():
            path = self._mark_path()
            return path


class AStarSimple(AStar):
    def __init__(self):
        super(AStarSimple,self).__init__()

    def navigate_simple(self,maze,start_pose,end_pose):
        maze = cv2.resize(maze,(self.config['map']['x']/4,self.config['map']['y']/4),interpolation=cv2.INTER_NEAREST)
        start_pose = (start_pose[0]/4,start_pose[1]/4)
        end_pose = (end_pose[0]/4,end_pose[1]/4)
        path = self.navigate(maze,start_pose,end_pose)
        if path is None:
            path = self.navigate(maze,start_pose,(end_pose[0],end_pose[1]+1))
            if path is None:
                path = self.navigate(maze,start_pose,(end_pose[0],end_pose[1]-1))
                if path is None:
                    path = self.navigate(maze,start_pose,(end_pose[0]+1,end_pose[1]))
                    if path is None:
                        path = self.navigate(maze,start_pose,(end_pose[0]-1,end_pose[1]))
        return path


if __name__ == '__main__':
    maze = np.array((
        (0, 0, 0, 0, 0, 0, 0),
        (0, 100, 100, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0),
    ))
    start = (0,0)
    end = (4,5)
    astar = AStarSimple()
    path = astar.navigate(maze,(start[0],start[1]),(end[0],end[1]))
    # astar = AStar(maze,start[0],start[1],end[0],end[1])
    print(path)
    v_map = np.copy(maze)

    plt.figure(1)
    plt.clf()
    v_map[start]=150
    v_map[end]=150
    for point in path:
        v_map[point] = 150

    plt.imshow(v_map, cmap='gray')
    plt.draw()
    plt.pause(10)
