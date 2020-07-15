#coding:utf8
import torch as t
import os
import time


class BasicModule(t.nn.Module):
    """
    封装了nn.Module,主要是提供了save和load两个方法
    """

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self))# 默认名字

    def load(self, path):
        """
        可加载指定路径的模型
        """
        # print(self.parameters())
        # self.load_state_dict(t.load(path))
        # print(self.parameters())
        # self.eval()
        # return
        raise NotImplemented

    def save(self,tp,id,name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        # if name is None:
        #     prefix = os.getcwd()+'/checkpoints/' + tp + '_' + id
        #     name = prefix + '.pth'
        # t.save(self.state_dict(), name)
        # return name
        raise NotImplemented

    def get_optimizer(self, lr, weight_decay):
        return t.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)