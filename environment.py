
import numpy as np
import time
import math
import itertools
import queue
import sys

from sqlalchemy import false, true

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

eng_ratio = 100
eg_eff = 0.5 
snr_th=3
gamma=0.5
SNR=30
N0 = 0.01
Ps = pow(10,SNR/10.0)*N0
Pr = Ps/eng_ratio


def give_channel():
    return pow(abs(1/math.sqrt(2)*(np.random.normal(0,1)+np.random.normal(0,1)*1j)),2)

def choose_channel(channels):
    return np.random.choice(channels)

def NormalizeData(data):
    return data / np.max(data)

class twohop_relay(tk.Tk, object):
    def __init__(self, rly_num, data_size, eng_size):
        super(twohop_relay, self).__init__()
        self.action_space = list(range(0,rly_num*rly_num))
        self.n_actions = len(self.action_space)
        self.rly_num = rly_num
        self.data_size = data_size
        self.eng_size = eng_size
        self.dis = 1
        self.ini_data = 1
        self.ini_eng = 1
        self.n_features = 4
        self.age = [queue.Queue(), queue.Queue(), queue.Queue()]
        self._build_system()
        #紀錄下OR的使用次數
        self.OR_his = 0

    def _build_system(self):
        self.data = list(itertools.repeat(self.ini_data, self.rly_num))
        self.eng = list(itertools.repeat(self.ini_eng, self.rly_num))
        self.random_channel =  np.array([give_channel() for _ in range(100)])
        self.maxchannel = np.max(self.random_channel)
        self.h = [choose_channel(NormalizeData(self.random_channel)) for _ in range(self.rly_num)]
        self.g = [choose_channel(NormalizeData(self.random_channel)) for _ in range(self.rly_num)]
        self.snr1 = [Ps*pow(abs(self.h[i]),2) for i in range(self.rly_num)]
        self.snr2 = [Pr*pow(abs(self.g[i]),2) for i in range(self.rly_num)]

    def sendOR_his(self):
        return self.OR_his
    
    def changedis(self, dist):
        self.dis = dist
    
    def reset(self):
        # return observation
        self.data = list(itertools.repeat(self.ini_data, self.rly_num))
        self.eng = list(itertools.repeat(self.ini_eng, self.rly_num))
        self.h = [choose_channel(self.random_channel) for _ in range(self.rly_num)]
        self.g = [choose_channel(self.random_channel) for _ in range(self.rly_num)]
        self.snr1 = [Ps*pow(abs(self.h[i]),2) for i in range(self.rly_num)]
        self.snr2 = [Pr*pow(abs(self.g[i]),2) for i in range(self.rly_num)]
        for i in range(self.rly_num):
            while not self.age[i].empty():
                self.age[i].get()
            self.age[i].put(0)
        self.OR_his=0
        return [self.h,self.g, self.data, self.eng]

    def Isvalidaction(self, action):
        recep = int(action/self.rly_num)
        trans = int(action%self.rly_num)

        if all(flag == 0 for flag in self.eng):
            return true
        elif (recep==trans) and (self.eng[trans]==0 ): # It's invalid state.
            return false
        elif (recep!=trans) and (self.data[recep]>=self.data_size-1 or self.eng[trans]==0 or self.data[trans]==0) : # It's invalid state.
            return false
        else:
            return true
    
    def ORchoosing(self):
        minsnr = []
        for r in range(self.rly_num):
            if self.eng[r]==0:
                continue
            minsnr.append(min(self.snr1[r], self.snr2[r]))
        rly = np.argmax(np.array(minsnr))

        return rly*self.rly_num+rly
    
    def choose(self, action):
        if self.Isvalidaction(action) == false:
            self.OR_his+=1
            action = self.ORchoosing()
        # print("my action", action)
        return action

    def step(self, action):
        cur_state = [self.h,self.g, self.data, self.eng]
        # print("current state ", cur_state)
        recep = int(action/self.rly_num)
        trans = int(action%self.rly_num)


        if all(flag == 0 for flag in self.eng):   # It's EES state
            for r in range(0,self.rly_num):  
                # *pow(dis, -2)
                self.eng[r] = self.eng[r]+ min(math.floor(self.h[r]*self.maxchannel*eg_eff*eng_ratio*pow(self.dis, -2)),self.eng_size)
            # print("It's EES.")
            reward=0
            for a in self.age:
                for _ in range(a.qsize()):
                    temp = a.get()+1
                    a.put(temp)
                    reward+=temp
            age = reward / sum(self.data)
            reward = -1/age
            self.render()
            # for a in self.age:
            #     print("Age ",a.queue)
            next_state = [self.h,self.g, self.data, self.eng]

        elif (recep==trans) and (self.eng[trans]==0 ): # It's invalid state
            # print("It's Invalid.")
            reward=0
            for a in self.age:
                for _ in range(a.qsize()):
                    temp = a.get()+1
                    a.put(temp)
                    reward+=temp
            age = reward / sum(self.data)
            reward = -1/age
            self.render()
            # for a in self.age:
            #     print("Age ",a.queue)
            next_state = [self.h,self.g, self.data, self.eng]
        
        elif (recep!=trans) and (self.data[recep]>=self.data_size-1 or self.eng[trans]==0 or self.data[trans]==0) :
            # print("It's Invalid.")
            reward=0
            for a in self.age:
                for _ in range(a.qsize()):
                    temp = a.get()+1
                    a.put(temp)
                    reward+=temp
            age = reward / sum(self.data)
            reward = -1/age
            self.render()
            # for a in self.age:
            #     print("Age ",a.queue)
            next_state = [self.h,self.g, self.data, self.eng]

            
        else:
            self.data[recep]+=1  # move agent
            self.data[trans]-=1
            for r in range(0,self.rly_num):  # transmit data to the reception reley, while the others charging
                if r != recep:
                    # *pow(dis, -2)
                    self.eng[r]=min(math.floor(self.eng[r]+self.h[r]*self.maxchannel*eg_eff*eng_ratio*pow(self.dis, -2)),self.eng_size)
            self.eng[trans]-=1
            self.render()
            # print("It's normal")
            # print("Recep ", recep, " Trans ", trans)

            # reward function
            if recep != trans:
                reward=0
                self.age[trans].get()
                for a in self.age:
                    for _ in range(a.qsize()):
                        temp = a.get()+1
                        a.put(temp)
                        reward+=temp
                age = reward / (sum(self.data)-1)
                reward = 1/age
                self.age[recep].put(0)
            else:
                reward=0
                if self.age[trans].empty():
                    for a in self.age:
                        for _ in range(a.qsize()):
                            temp = a.get()+1
                            a.put(temp)
                            reward+=temp
                    age = reward / (sum(self.data)-1)
                else:
                    self.age[trans].get()
                    for a in self.age:
                        for _ in range(a.qsize()):
                            temp = a.get()+1
                            a.put(temp)
                            reward+=temp
                    age = reward / (sum(self.data)-1)
                    self.age[recep].put(0)
                
                if age == 0:
                    age=0.1
                reward = 1/age
            # print("age = ",age)
            # for a in self.age:
            #     print("Age ",a.queue)
            next_state = [self.h,self.g, self.data, self.eng]
        # print("next state ", next_state)
        # input("Check Reward!")
        return next_state, reward, age

    def render(self):
        # time.sleep(0.01)
        self.h = [choose_channel(NormalizeData(self.random_channel)) for _ in range(self.rly_num)]
        self.g = [choose_channel(NormalizeData(self.random_channel)) for _ in range(self.rly_num)]
        self.snr1 = [Ps*pow(abs(self.h[i]),2) for i in range(self.rly_num)]
        self.snr2 = [Pr*pow(abs(self.g[i]),2) for i in range(self.rly_num)]