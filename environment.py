
import numpy as np
import time
import math
import sys

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


snr_th=3
gamma=0.5



class twohop_relay(tk.Tk, object):
    def __init__(self, rly_num, data_size, eng_size, snr_min, snr_max, snr_interval):
        super(twohop_relay, self).__init__()
        self.action_space = list(range(0,rly_num*rly_num+1))
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
        self.data = [self.ini_data for _ in self.rly_num]
        self.eng = [self.ini_eng for _ in self.rly_num]
        self.h = [1/math.sqrt(2)*(np.random.normal(0,1)+np.random.normal(0,1)*1j) for _ in self.rly_num]
        self.g = [1/math.sqrt(2)*(np.random.normal(0,1)+np.random.normal(0,1)*1j) for _ in self.rly_num]

    def reset(self):
        # return observation
        self.data = [self.ini_data for _ in self.rly_num]
        self.eng = [self.ini_eng for _ in self.rly_num]
        self.h = [1/math.sqrt(2)*(np.random.normal(0,1)+np.random.normal(0,1)*1j) for _ in self.rly_num]
        self.g = [1/math.sqrt(2)*(np.random.normal(0,1)+np.random.normal(0,1)*1j) for _ in self.rly_num]
        return [self.data, self.eng, self.h,self.g]

    def step(self, action):
        
        recep = action/self.rly_num
        trans = action%self.rly_num

        # reward function
        reward = 5*0.2*(self.h[recep]+1)*0.2*(self.g[trans]+1)/(5*0.2*(self.h[recep]+1)+0.2*(self.g[trans]+1)+1)
        reward = 0.5*math.log2(1+reward)
        
        self.data[recep]+=1  # move agent
        self.data[trans]-=1
        for r in range(0,self.rly_num):  # transmit data to the reception reley, while the others charging
            if r.id != recep:
                self.eng[r]+=min(pow(abs(self.h[r]),2)*eg_eff*eng_ratio,self.eng_size)
        self.eng[trans]-=1

        next_state = [self.data, self.eng, self.h,self.g]
        return next_state, reward

    def render(self):
        # time.sleep(0.01)
        self.h = [1/math.sqrt(2)*(np.random.normal(0,1)+np.random.normal(0,1)*1j) for _ in self.rly_num]
        self.g = [1/math.sqrt(2)*(np.random.normal(0,1)+np.random.normal(0,1)*1j) for _ in self.rly_num]