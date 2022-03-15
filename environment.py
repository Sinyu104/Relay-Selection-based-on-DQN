
import numpy as np
import time
import math
import itertools
import sys

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

eng_ratio = 5
eg_eff = 0.5 
snr_th=3
gamma=0.5


def give_channel():
    return pow(abs(1/math.sqrt(2)*(np.random.normal(0,1)+np.random.normal(0,1)*1j)),2)

def choose_channel(channels):
    return np.random.choice(channels)

def NormalizeData(data):
    return data / np.max(data)

class twohop_relay(tk.Tk, object):
    def __init__(self, rly_num, data_size, eng_size, snr_min, snr_max, snr_interval):
        super(twohop_relay, self).__init__()
        self.action_space = list(range(0,rly_num*rly_num))
        self.n_actions = len(self.action_space)
        self.rly_num = rly_num
        self.data_size = data_size
        self.eng_size = eng_size
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.snr_interval = snr_interval
        self.ini_data = 1
        self.ini_eng = 1
        self.n_features = 4
        self._build_system()

    def _build_system(self):
        self.data = list(itertools.repeat(self.ini_data, self.rly_num))
        self.eng = list(itertools.repeat(self.ini_eng, self.rly_num))
        self.random_channel =  NormalizeData(np.array([give_channel() for _ in range(10)]))
        self.h = [choose_channel(self.random_channel) for _ in range(self.rly_num)]
        self.g = [choose_channel(self.random_channel) for _ in range(self.rly_num)]
        

    
    def reset(self):
        # return observation
        self.data = list(itertools.repeat(self.ini_data, self.rly_num))
        self.eng = list(itertools.repeat(self.ini_eng, self.rly_num))
        self.h = [choose_channel(self.random_channel) for _ in range(self.rly_num)]
        self.g = [choose_channel(self.random_channel) for _ in range(self.rly_num)]
        return [self.h,self.g, self.data, self.eng]

    def step(self, action):
        cur_state = [self.h,self.g, self.data, self.eng]
        recep = int(action/self.rly_num)
        trans = action%self.rly_num
        if self.data[recep]>=self.data_size-1 or self.eng[trans]==0 or self.data[trans]==0: # It's invalid state
            reward = -1
            next_state = cur_state

        elif all(flag == 0 for flag in self.eng):   # It's EES state
            reward = -1
            for r in range(0,self.rly_num):  # transmit data to the reception reley, while the others charging
                self.eng[r]+=min(math.floor(self.h[r]*0.4*eg_eff*eng_ratio),self.eng_size)
            next_state = cur_state
        else:
            self.data[recep]+=1  # move agent
            self.data[trans]-=1
            for r in range(0,self.rly_num):  # transmit data to the reception reley, while the others charging
                if r != recep:
                    self.eng[r]=min(math.floor(self.eng[r]+self.h[r]*0.4*eg_eff*eng_ratio),self.eng_size)
            self.eng[trans]-=1
            self.render()

            # reward function

            reward = 5*0.2*(self.h[recep]*0.4+1)*0.2*(self.g[trans]*0.4+1)/(5*0.2*(self.h[recep]*0.4+1)+0.2*(self.g[trans]*0.4+1)+1)
            reward = 0.5*math.log2(1+reward)
            next_state = [self.h,self.g, self.data, self.eng]
        
        
        return next_state, reward

    def render(self):
        # time.sleep(0.01)
        self.h = [choose_channel(self.random_channel) for _ in range(self.rly_num)]
        self.g = [choose_channel(self.random_channel) for _ in range(self.rly_num)]