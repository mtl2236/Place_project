from gym.spaces import Discrete
import torch
import torch.nn as nn
import numpy as np
from gym.utils import seeding
import os
import sys
import logging
import math
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from rnd import RNDModel
import torch.optim as optim
from Valid_list_gen import Coordinates_transform, get_range, write_best_result  

np.set_printoptions(threshold=np.inf)
rnd = RNDModel((1, 1, 84, 84), 32*32)
forward_mse = nn.MSELoss(reduction='none')
optimizer = optim.Adam(rnd.predictor.parameters(), lr=5e-6)


def compute_intrinsic_reward(rnd, next_obs):
    
    target_next_feature = rnd.target(next_obs)
    predict_next_feature = rnd.predictor(next_obs)

    forward_loss = forward_mse(predict_next_feature, target_next_feature).mean(-1)
    intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2
    optimizer.zero_grad()
    forward_loss.backward()

    return intrinsic_reward.item()/100

"""Guangxi"""
def is_inrange(x, y, grid_size):
    if -1 < x < grid_size and -1 < y < grid_size:
        return True
    return False
"""Guangxi"""    
def is_valid(ob, x, dx, y, dy, grid_size):                        #judge is overlap   矩形包围的内部各点不处于已经被占据的（赋值为1）
    for u in range(dx):
        for s in range(dy):
            """Guangxi"""
            if ((False==is_inrange(x+u,y+s,grid_size))or(ob[x+u, y+s]>0)):
                return False
    return True

"""Guangxi"""
def search(ob, x, dx, y, dy, depth, n, grid_size):
    """Guangxi"""
    if(True==is_valid(ob, x, dx, y, dy, grid_size)):
        return x, y
    if depth > 7:
        return -1, -1
        """Guangxi"""    
    elif (True==is_valid(ob, x-1, dx, y, dy, grid_size)):
        return x-1, y
        """Guangxi"""    
    elif (True==is_valid(ob, x+1,dx, y, dy, grid_size)):
        return x+1, y
        """Guangxi"""
    elif (True==is_valid(ob, x, dx, y-1, dy, grid_size)):
        return x, y-1
        """Guangxi"""
    elif (True==is_valid(ob, x, dx, y+1, dy, grid_size)):
        return x, y+1
    else:
        """Guangxi"""
        return search(ob, x-1, dx, y-1, dy, depth+1, n, grid_size)

"""Guangxi"""
def find(ob, n, dx, dy, grid_size):
    center = [n//2, n//2]
    for i in range(n):
        for j in range(i):
            """Guangxi"""
            if True==is_valid(ob, center[0]-j, dx, center[1]-(i-j), dy, grid_size) :
                return center[0]-j, center[1]-(i-j)
            """Guangxi"""
            if True==is_valid(ob, center[0]-j, dx, center[1]+(i-j), dy, grid_size):
                return center[0]-j, center[1]+(i-j)
            """Guangxi"""
            if True==is_valid(ob, center[0]+j, dx, center[1]-(i-j), dy, grid_size):
                return center[0]+j, center[1]-(i-j)
            """Guangxi"""
            if True==is_valid(ob, center[0]+j, dx, center[1]+(i-j), dy, grid_size):
                return center[0]+j, center[1]+(i-j)
    return -1,-1
"""Guangxi"""
def cal_re(r, x, grid_size_n):
    """Guangxi"""
    Macro_center_point_list, Ports_of_macro_list = Coordinates_transform(r, grid_size_n)
    #print(Ports_of_macro_list)
    wl = 0
    #con = np.zeros((32, 32))
    # areatop = 0
    # areadown = 31
    # arealeft = 31
    # arearight = 0
    for net in x:
        left = 1000000
        right = 0
        up = 1000000
        down = 0
        for i in range(len(net)):
            left = min(left, Ports_of_macro_list[net[i][0]][net[i][1]-1][0])
            right = max(right, Ports_of_macro_list[net[i][0]][net[i][1]-1][0])
            up = min(up, Ports_of_macro_list[net[i][0]][net[i][1]-1][1])
            down = max(down, Ports_of_macro_list[net[i][0]][net[i][1]-1][1])
            # if up<areadown:
                # areadown=up
            # if down>areatop:
                # areatop=down
            # if left<arealeft:
                # arealeft=left
            # if right>arearight:
                # arearight=right
        wn = right-left
        hn = down-up
        #dn = (wn+hn) / (wn*hn)
        
        #con[up:down+1, left:right+1] += dn
        wl += wn + hn

    #Valid_final_list, Util_macro, Util_area_macro, Macro_center_point_list = Coordinates_transform(r)
    #con = list(con.flatten())
    #area=int(areatop-areadown)*int(arearight-arealeft)
    #con.sort(reverse=True)
    #print('down',up,'up',down,'left',left,'right',right)
    #print('area',area,'con',con[:32])
    #(-np.mean(con[:32]) - (wl-34000)*0.1)*0.2
    print('hpwl ',wl,'\n')
    #print('cong*0.2 ',np.mean(con[:32])*0.2,'\n')
    #print('Util_macro ', Util_macro,'\n')
    #print('Util_area_macro ',Util_area_macro,'\n' )
    #print('Final reward', (-np.mean(con[:32]) - (wl-34000)*0.1)*0.2, '\n')
    #print('Final reward ', wl, '\n')
    #print('reward',10000*Util_macro*Util_area_macro)
    #print('final reward',10000*Util_macro,'\n')
    #print('reward',(-np.mean(con[:32]) - wl*0.1)*0.2 + 10000*Util_macro*Util_area_macro)
    #return (-np.mean(con[:32]) - wl*0.1)*0.2 + 10000*Util_macro*Util_area_macro+700, wl, np.mean(con[:32]), Util_macro, Util_area_macro
    #return 10000*Util_macro*Util_area_macro+700
    return 10000000-wl, Macro_center_point_list
# num_cell=710
class Placememt():
    """Guangxi"""
    def __init__(self, grid_size=32, num_cell=50, obs_space = 84, case_id=1):
        self.n = grid_size
        self.steps = num_cell
        self.action_space = Discrete(self.n * self.n)
        """Guangxi"""
        self.obs_space = (1, obs_space, obs_space)
        """Guangxi"""
        self.obs_number = obs_space
        self.obs = torch.zeros((1, 1, self.n, self.n))
        self.results = []
        #self.best_results=[]
        self.case_id=case_id
        self.best = -500
        self.f = open("./result/result.txt", 'w')

        f = open("./data/n_edges.dat", "r")
        for line in f:
            self.net = eval(line)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        self.obs = torch.zeros((1, 1, self.n, self.n))
        return self.obs

    def transform(self, x):
        """Guangxi"""
        up = nn.Upsample(size=self.obs_number, mode='bilinear', align_corners=False)
        return up(x)*255
    """Guangxi"""
    def step(self, action, grid_size_n):
        #print('action',action,'\n')
        x = action // self.n
        y = action % self.n
        """Guangxi"""
        dx=math.ceil(get_range(len(self.results),self.steps, grid_size_n)[0])    #current macro x length in 32*32 space  self.results=macro id
        dy=math.ceil(get_range(len(self.results),self.steps, grid_size_n)[1])    #current macro y length in 32*32 space
        """Guangxi"""
        x, y = search(self.obs[0, 0], x, dx, y, dy, 0, self.n, self.n)
        if(x == -1 or y == -1):   #8 times of serach failed
            """Guangxi"""
            x, y = find(self.obs[0, 0], self.n, dx, dy, self.n)
        if(x == -1 or y == -1):  #no place for current macro end this episode
            print('no place for current macro end this episode\n')
            done = True
            reward=-500+len(self.results)/self.steps 
            print('reward not complete',reward,'\n')
            obs = self.transform(self.obs)
            """Guangxi"""
            Macro_center_point_list=Coordinates_transform(self.results, grid_size_n)[0]
            if reward > self.best:
                self.best = reward
                write_best_result(Macro_center_point_list, self.case_id)   #update output txt
                #self.f.write('\n')
                #self.f.write(str(self.results))
                #self.f.write('\n')
            self.f.write(str(reward))
            self.f.write('\n')
            self.results = []
            return obs, done, torch.FloatTensor([[reward]])
        else:
            """Guangxi"""
            for u in range(x,min(x+dx,self.n)):
                """Guangxi"""
                for t in range(y,min(y+dy,self.n)):
                    self.obs[0, 0, u, t] = 1 
            self.results.append([int(x), int(y)])
            obs = self.transform(self.obs)
        #print(obs.shape)
        if len(self.results) == self.steps:
            print('complete!\n')
            done = True
            #print('result ',self.results, '\n')
            """Guangxi"""
            #reward,wirelength,congestion, Util_macro, Util_area_macro  = cal_re(self.results, self.net)
            reward,Macro_center_point_list=cal_re(self.results, self.net, grid_size_n)
            print('complete reward',reward, '\n')
            if reward > self.best:
                self.best = reward
                #self.best_results=self.results
                #print(self.best_results)
                write_best_result(Macro_center_point_list, self.case_id)   #update output txt
                #self.f.write(str(self.obs))
                #self.f.write('\n')
                #self.f.write(str(self.results))
                #self.f.write('\n')
            self.f.write(str(reward))
            self.f.write('\n')
                #self.f.write('hpwl')
                #self.f.write(str(wirelength))
                #self.f.write('\n')
                #self.f.write('congestion')
                #self.f.write(str(congestion))
                #self.f.write('\n')
                #self.f.write('Util_macro')
                #self.f.write(str(Util_macro))
                #self.f.write('\n')
                #self.f.write('Util_area_macro')
                #self.f.write(str(Util_area_macro))
            self.results = []
        else:
            done = False
            reward = compute_intrinsic_reward(rnd, obs / 255.0)
            #extra_reward=Coordinates_transform(self.results)[1]
            #print('extra_reward',extra_reward,'\n')
            # if(extra_reward<1):
                # reward=reward-1000
            # else:
                # reward=reward
            #reward=0
            #print('rnd',reward,'\n')
        return obs, done, torch.FloatTensor([[reward]])

"""Guangxi"""
def place_envs(grid_size = 32, num_cell = 50, obs_space = 84, case_id=1):
    return Placememt(grid_size, num_cell, obs_space, case_id)
