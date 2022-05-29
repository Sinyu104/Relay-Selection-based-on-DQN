
from types import AsyncGeneratorType
import numpy as np
import time
import math
import itertools
import queue
import sys
import DBRS as db
import SAR_LAT as SRLT

# from sqlalchemy import false, true

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


def choose_channel(channels):
    return np.random.choice(channels)

def NormalizeData(data):
    return data / np.max(data)

def give_channel(dis):
    channel = []
    for d in dis:
        for _ in range(100000):
            channel.append(pow(abs(1/math.sqrt(2)*(np.random.normal(0,1)+np.random.normal(0,1)*1j)),2)*pow(d, -2))
    max_channel = np.amax(channel)
    channel[:] = [x / max_channel for x in channel]
    return channel, max_channel

class twohop_relay(tk.Tk, object):
    def __init__(self, rly_num, data_size, eng_size):
        super(twohop_relay, self).__init__()
        self.action_space = list(range(0,rly_num*rly_num))
        self.n_actions = len(self.action_space)
        self.rly_num = rly_num
        self.data_size = data_size
        self.eng_size = eng_size
        self.ini_data = 1
        self.ini_eng = 1
        self.n_features = 5
        self.dis = [5]
        self.age = [queue.Queue() for _ in range(self.rly_num)]
        self.cur_age = [0 for _ in range(self.rly_num)]
        self._build_system()
        #紀錄下OR的使用次數
        self.OR_his = 0

    def _build_system(self):
        self.data = list(itertools.repeat(self.ini_data, self.rly_num))
        self.eng = list(itertools.repeat(self.ini_eng, self.rly_num))
        self.random_channel , self.maxchannel=  give_channel(self.dis)
        self.h = [choose_channel(self.random_channel) for _ in range(self.rly_num)]
        self.g = [choose_channel(self.random_channel) for _ in range(self.rly_num)]
        self.snr1 = [Ps*pow(abs(self.h[i]*self.maxchannel),2)/N0 for i in range(self.rly_num)]
        self.snr2 = [Pr*pow(abs(self.g[i]*self.maxchannel),2)/N0 for i in range(self.rly_num)]


    def sendOR_his(self):
        return self.OR_his
    
    def returnstate(self):
        return [self.h, self.g, self.data, self.eng, self.age[0].queue, self.age[1].queue, self.age[2].queue, self.cur_age, self.maxchannel]
    
    def reset(self):
        # return observation
        self.data = list(itertools.repeat(self.ini_data, self.rly_num))
        self.eng = list(itertools.repeat(self.ini_eng, self.rly_num))
        self.h = [choose_channel(self.random_channel) for _ in range(self.rly_num)]
        self.g = [choose_channel(self.random_channel) for _ in range(self.rly_num)]
        self.snr1 = [Ps*pow(abs(self.h[i]*self.maxchannel),2)/N0 for i in range(self.rly_num)]
        self.snr2 = [Pr*pow(abs(self.g[i]*self.maxchannel),2)/N0 for i in range(self.rly_num)]
        for i in range(self.rly_num):
            while not self.age[i].empty():
                self.age[i].get()
            self.age[i].put(0)
        self.OR_his=0
        self.cur_age = [0 for _ in range(self.rly_num)]
        self.data_normalize = [x / (self.data_size-1) for x in self.data]
        self.eng_normalize = [x / (self.eng_size) for x in self.eng]
        self.cur_age_normalize = self.cur_age.copy()
        return [self.h,self.g, self.data_normalize, self.eng_normalize, self.cur_age_normalize]
    

    def Isvalidaction(self, action):
        recep = int(action/self.rly_num)
        trans = int(action%self.rly_num)

        if all(flag == 0 for flag in self.eng):
            return True
        elif (recep==trans) and (self.eng[trans]==0 ): # It's invalid state.
            return False
        elif (recep!=trans) and (self.data[recep]>=self.data_size-1 or self.eng[trans]==0 or self.data[trans]==0) : # It's invalid state.
            return False
        else:
            return True
    
    def ORchoosing(self):
        minsnr = []
        for r in range(self.rly_num):
            if self.eng[r]==0:
                minsnr.append(-1)
                continue
            minsnr.append(min(self.snr1[r], self.snr2[r]))
        rly = np.argmax(np.array(minsnr))

        return rly*self.rly_num+rly
    
    def choose(self, action):
        if self.Isvalidaction(action) == False:
            action = SRLT.SAR_LAT(rly_num=self.rly_num, data_size=self.data_size, data_bf=self.data, eng_bf=self.eng, cur_age = self.cur_age)
        if action < 0:
            action = self.DBRS_OR()
        return action
    
    def DBRS_OR(self):
        if all(flag == 0 for flag in self.eng):
            return np.random.randint(0, self.n_actions)
        action = db.DBRS(rly_num=self.rly_num, data_size=self.data_size, snr1=self.snr1, snr2=self.snr2, data_bf=self.data, eng_bf=self.eng)
        if action < 0:
            action = self.ORchoosing()
        return action
    
    def SARLAT(self):
        if all(flag == 0 for flag in self.eng):
            return np.random.randint(0, self.n_actions)
        action = SRLT.SAR_LAT(rly_num=self.rly_num, data_size=self.data_size, data_bf=self.data, eng_bf=self.eng, cur_age = self.cur_age)
        if action < 0:
            action = self.ORchoosing()
        return action
        

    def step(self, action):
        # cur_state = [self.h,self.g, self.data, self.eng, self.cur_age]
        # print("current state ", cur_state)
        recep = int(action/self.rly_num)
        trans = int(action%self.rly_num)
        # print("Recep: ", recep, "Trans: ", trans)
        # origin_age = 0
        # for a in self.age:
        #     for _ in range(a.qsize()):
        #         temp = a.get()
        #         origin_age+=temp
        #         a.put(temp)
        # print("origin_age: ", origin_age)


        if all(flag == 0 for flag in self.eng):   # It's EES state
            for r in range(0,self.rly_num):
                self.eng[r] = self.eng[r]+ min(math.floor(self.h[r]*self.maxchannel*eg_eff*eng_ratio),self.eng_size)
            # print("It's EES.")
            age=0
            for a in self.age:
                for _ in range(a.qsize()):
                    temp = a.get()+1
                    a.put(temp)
                    age+=temp
            reward = -10
            age = -1
            # reward = 1/(origin_age - reward)
            # sum(self.eng)/self.eng_size/self.rly_num
            
            self.render()
            # reward = (origin_age - reward)/10
            # for a in self.age:
            #     print("Age ",a.queue)
            # print("reward ", reward)
            next_state = [self.h,self.g, self.data_normalize, self.eng_normalize, self.cur_age_normalize]

        elif (recep==trans) and (self.eng[trans]==0 ): # It's invalid state
            # print("It's Invalid.")
            age=0
            for a in self.age:
                for _ in range(a.qsize()):
                    temp = a.get()+1
                    a.put(temp)
                    age+=temp

            reward = -10
            age = -1
            # reward = 1/(origin_age - reward)
            
            self.render()
            # reward = (origin_age - reward)/10
            # for a in self.age:
            #     print("Age ",a.queue)
            # print("reward ", reward)
            next_state = [self.h,self.g, self.data_normalize, self.eng_normalize, self.cur_age_normalize]
        
        elif (recep!=trans) and (self.data[recep]>=self.data_size-1 or self.eng[trans]==0 or self.data[trans]==0) :
            # print("It's Invalid.")
            age=0
            for a in self.age:
                for _ in range(a.qsize()):
                    temp = a.get()+1
                    a.put(temp)
                    age+=temp
            reward = -10
            age = -1
            # reward = 1/(origin_age - reward)
            # reward = (origin_age - reward)/10
            self.render()
            # for a in self.age:
            #     print("Age ",a.queue)
            # print("reward ", reward)
            next_state = [self.h,self.g, self.data_normalize, self.eng_normalize, self.cur_age_normalize]

            
        else:
            self.data[recep]+=1  # move agent
            self.data[trans]-=1
            for r in range(0,self.rly_num):  # transmit data to the reception reley, while the others charging
                if r != recep:
                    self.eng[r]=min(math.floor(self.eng[r]+self.h[r]*self.maxchannel*eg_eff*eng_ratio),self.eng_size)

            self.eng[trans]-=1
            # print("It's normal")
            # print("Recep ", recep, " Trans ", trans)
            

            # reward function
            if recep != trans:
                age=0
                reward = self.age[trans].get()
                for a in self.age:
                    for _ in range(a.qsize()):
                        temp = a.get()+1
                        a.put(temp)
                        age+=temp
                
                age = reward 
                reward = -1*reward
                self.age[recep].put(0)
            else:
                age=0
                if self.age[trans].empty():
                    for a in self.age:
                        for _ in range(a.qsize()):
                            temp = a.get()+1
                            a.put(temp)
                            age+=temp
                    reward = 0
                    age = 0
                else:
                    reward = self.age[trans].get()
                    for a in self.age:
                        for _ in range(a.qsize()):
                            temp = a.get()+1
                            a.put(temp)
                            age+=temp
                    age = reward
                    reward = 0
                    self.age[recep].put(0)
                
            # for a in self.age:
            #     print("Age ",a.queue)
            # print("reward ", reward)
            self.render()
            next_state = [self.h,self.g, self.data_normalize, self.eng_normalize, self.cur_age_normalize]
            # print("next state ", next_state)
        return next_state, reward, age
    

    def render(self):
        self.h = [choose_channel(self.random_channel) for _ in range(self.rly_num)]
        self.g = [choose_channel(self.random_channel) for _ in range(self.rly_num)]
        self.snr1 = [Ps*pow(abs(self.h[i]*self.maxchannel),2)/N0 for i in range(self.rly_num)]
        self.snr2 = [Pr*pow(abs(self.g[i]*self.maxchannel),2)/N0 for i in range(self.rly_num)]
        for i in range(self.rly_num):
            if self.age[i].empty():
                self.cur_age[i]=0
            else:
                self.cur_age[i]=list(self.age[i].queue)[0]
        
        self.data_normalize = [x / (self.data_size-1) for x in self.data]
        self.eng_normalize = [x / (self.eng_size) for x in self.eng]
        for i in range(self.rly_num):
            if np.max(self.cur_age)==0:
                self.cur_age_normalize[i]=self.cur_age
            else:
                self.cur_age_normalize[i] = self.cur_age[i]/np.max(self.cur_age)