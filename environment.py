#from asyncio.windows_events import NULL
from pyexpat import model
from types import AsyncGeneratorType
from statistics import mean, variance
import numpy as np
import time
import math
import itertools
import queue
import sys
import random
from numpy import Inf
import DBRS as db
import SAR_LAT as SRLT
import Max_link as MLink

# from sqlalchemy import false, true

# if sys.version_info.major == 2:
#     import Tkinter as tk
# else:
#     import tkinter as tk

eng_ratio = 1000
eg_eff = 0.5 
snr_th=3
gamma=0.5
SNR=70
N0 = 0.01
Ps = pow(10,SNR/10.0)*N0
Pr = Ps/eng_ratio



def NormalizeData(data):
    return data / np.max(data)

def give_channel(dis):
    channel = []
    for _ in range(1000000):
        channel.append(pow(abs(1/math.sqrt(2)/36*(np.random.normal(0,1)+np.random.normal(0,1)*1j)),2))
    max_channel = np.amax(channel)
    channel[:] = [x / max_channel for x in channel]
    return channel, max_channel

class twohop_relay:
    def __init__(self, rly_num, data_size, eng_size):
        self.action_space = list(range(0,rly_num))
        self.n_actions = 2*len(self.action_space)
        self.rly_num = rly_num
        self.data_size = data_size
        self.eng_size = eng_size
        self.ini_data = 0
        self.ini_eng = 1
        self.n_features = 6
        self.arrival = 0.3
        self.interarrival_times = np.random.default_rng().exponential(scale=1/self.arrival, size=500000)
        self.arrival_times = np.floor(np.cumsum(self.interarrival_times))
        self.dis = 6
        self.age = [queue.Queue() for _ in range(self.rly_num)]
        self.snr_th = 3
        self._build_system()
        #紀錄下OR的使用次數
        self.OR_his = 0
        self.DBRS_OR_number = 0

    def _build_system(self):
        self.data = list(itertools.repeat(self.ini_data, self.rly_num))
        self.eng = list(itertools.repeat(self.ini_eng, self.rly_num))
        self.random_channel , self.maxchannel=  give_channel(self.dis)
        self.h = [self.random_channel[random.randrange(0, len(self.random_channel))] for _ in range(self.rly_num)]
        self.g = [self.random_channel[random.randrange(0, len(self.random_channel))] for _ in range(self.rly_num)]
        self.snr1 = [Ps*self.h[i]*self.maxchannel/N0 for i in range(self.rly_num)]
        self.snr2 = [Pr*self.g[i]*self.maxchannel/N0 for i in range(self.rly_num)]
        self.h_ = [0 for _ in range(self.rly_num)]
        self.g_ = [0 for _ in range(self.rly_num)]
        self.rela_age = [0 for _ in range(self.rly_num)]
        self.pkg = [1 for _ in range(self.rly_num)]

    def sendOR_his(self):
        return self.OR_his
    
    def returnstate(self):
        return [self.h, self.g, self.data, self.eng, self.age[0].queue, self.age[1].queue, self.age[2].queue, self.snr1, self.snr2]
    
    def channel_refresh(self):
        self.random_channel , self.maxchannel=  give_channel(self.dis)
    
    def reset(self):
        # return observation
        self.data = list(itertools.repeat(self.ini_data, self.rly_num))
        self.eng = list(itertools.repeat(self.ini_eng, self.rly_num))
        self.h = [self.random_channel[random.randrange(0, len(self.random_channel))] for _ in range(self.rly_num)]
        self.g = [self.random_channel[random.randrange(0, len(self.random_channel))] for _ in range(self.rly_num)]
        self.snr1 = [Ps*self.h[i]*self.maxchannel/N0 for i in range(self.rly_num)]
        self.snr2 = [Pr*self.g[i]*self.maxchannel/N0 for i in range(self.rly_num)]
        for r in range(0,self.rly_num):
            if self.snr1[r]>=self.snr_th: 
                self.h_[r] = min(math.floor(self.h[r]*self.maxchannel*eg_eff*eng_ratio*10)/10.0,self.eng_size)
            else:
                self.h_[r] = -1
        self.g_ = [1 if s2>self.snr_th else 0 for s2 in self.snr2]
        for i in range(self.rly_num):
            while not self.age[i].empty():
                self.age[i].get()
            # self.age[i].put(0)
        self.OR_his=0
        self.for_age = [0 for _ in range(self.rly_num)]
        self.back_age = [0 for _ in range(self.rly_num)]
        self.rela_age = [0 for _ in range(self.rly_num)]
        self.data_normalize = [x / (self.data_size) for x in self.data]
        self.eng_normalize = [x / (self.eng_size) for x in self.eng]
        self.for_age_normalize = [0 for _ in range(self.rly_num)]
        # print("DBRS OR number", self.DBRS_OR_number)
        self.DBRS_OR_number = 0
        self.current_age = 0
        self.interarrival_times = np.random.default_rng().exponential(scale=1/self.arrival, size=500000)
        self.arrival_times = np.floor(np.cumsum(self.interarrival_times))
        self.rela_age = [(self.age[r].queue[0]-self.current_age)/100 if not self.age[r].empty() else 0 for r in range(self.rly_num)]
        self.pkg = [1 for _ in range(self.rly_num)]
        return [self.h_,self.g_,self.data_normalize,self.eng_normalize,self.rela_age,self.pkg]
    
    def IfArrive(self, time):
        if any(map(len, np.where(self.arrival_times == time))):
            return True
        else:
            return False
    
    def update_current_age(self, cur_age):
        self.current_age = cur_age

    def availible_action(self, mode):
        avai_action = []
        if mode == 1: 
            for r in range(self.rly_num):
                if self.snr1[r]>=snr_th:
                    avai_action.append(r)
        else:
            for r in range(self.rly_num):
                if self.snr2[r]>=snr_th and self.data[r]>0 and self.eng[r]>=1.0 and self.rela_age[r]>0:
                    avai_action.append(self.rly_num+r)
                else:
                    pass
        # print("availible action ", avai_action)
        return avai_action


    def Isvalidaction(self, mode, action):
        recep = int(action/(self.rly_num))
        trans = int(action%(self.rly_num))

        if all(flag == 0 for flag in self.eng):
            return True
        elif mode == 1 :
            if recep==trans:
                if self.eng[recep]!=0 and self.snr2[recep]>self.snr_th:
                    return True
                else:
                    return False
            else:
                if self.data[recep]==self.data_size-1 or self.eng[trans]==0 or self.data[trans]==0 or self.snr1[recep]<self.snr_th or self.snr2[trans]<self.snr_th:
                    return False
                else:
                    return True
        else:
            if self.data[trans]== 0 or self.eng[trans]==0 or self.snr2[trans]<self.snr_th:
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

        return rly*(self.rly_num)+rly
    
    def choose(self, mode, action):
        # recep = int(action/(self.rly_num))
        # trans = int(action%(self.rly_num))
        if self.Isvalidaction(mode, action) == False:
            action = self.ORchoosing()
        # if mode == 1:
        #     if self.snr1[recep] <self.snr_th:
        #         recep = np.argmax(self.snr1) 
        return action
    
    def DBRS_OR(self):
        if all(flag == 0 for flag in self.eng):
            return np.random.randint(0, self.n_actions)
        action = db.DBRS(rly_num=self.rly_num, data_size=self.data_size, snr1=self.snr1, snr2=self.snr2, data_bf=self.data, eng_bf=self.eng)
        if action < 0:
            self.DBRS_OR_number +=1
            action = self.ORchoosing()
        return action
    
    def SARLAT(self):
        if all(flag == 0 for flag in self.eng):
            return np.random.randint(0, self.n_actions)
        action = SRLT.SAR_LAT(rly_num=self.rly_num, snr1 = self.snr1, snr2 = self.snr2, age=self.age, data_bf=self.data, eng_bf=self.eng)
        if action < 0:
            action = self.ORchoosing()
        return action
    
    def Maxlink(self):
        [action, rly] = MLink.Max_link(rly_num=self.rly_num, h=self.h, g = self.g, data_bf=self.data, eng_bf=self.eng)
        return [action, rly]
        

    def step(self, mode, action, time, Age, next_mode):
        # cur_state = [self.h_,self.g_, self.data, self.eng, self.age[0].queue, self.age[1].queue,self.age[2].queue, self.pkg, self.rela_age]
        # print("current state ", cur_state)
        # print("Time ", time)
        # print("current age ", self.current_age)
        act = int(action/self.rly_num)
        rly = int(action%self.rly_num)
        # print("Action: ", act, "Relay: ", rly)
        # input("Pause")
 
        if act == 0: 
            if mode==1 and self.snr1[rly]<self.snr_th:
                reward = -5

            # elif mode==1 and recep==trans and self.eng[recep]!=0 and self.snr2[trans]>= self.snr_th:
            #     self.data[recep]+=1  # move agent
            #     for r in range(0,self.rly_num):  # transmit data to the reception reley, while the others charging
            #         if r != recep:
            #             self.eng[r]=min(math.floor(self.eng[r]+self.h[r]*self.maxchannel*eg_eff*eng_ratio),self.eng_size)
            #     self.age[recep].put(time)

            elif mode==1 and self.data[rly]!=self.data_size:
                self.data[rly]+=1  # move agent
                for r in range(0,self.rly_num):  # transmit data to the reception reley, while the others charging
                    if r != rly:
                        self.eng[r]=min(math.floor((self.eng[r]+self.h[r]*self.maxchannel*eg_eff*eng_ratio)*10)/10.0,self.eng_size)
                self.age[rly].put(time)


            elif mode==1 and self.data[rly]==self.data_size:
                for r in range(0,self.rly_num):  # transmit data to the reception reley, while the others charging
                    if r != rly:
                        self.eng[r]=min(math.floor((self.eng[r]+self.h[r]*self.maxchannel*eg_eff*eng_ratio)*10)/10.0,self.eng_size)
                self.age[rly].put(time)
                self.age[rly].get()
                
            else :
                pass
            age = -1
            self.render()
            reward = sum(self.rela_age)    
            
        else:
            for i in range(self.rly_num):
                if self.age[i].empty():
                    self.for_age[i]=0
                    self.back_age[i] = 0
                else:
                    self.for_age[i]=list(self.age[i].queue)[0]
                    self.back_age[i]=list(self.age[i].queue)[-1]
                    
                
            if self.data[rly]!=0 and self.eng[rly]>=1.0 and self.snr2[rly]>=self.snr_th:
                self.data[rly]-=1
                self.eng[rly]-=1.0
                age = self.age[rly].get()
                self.render()
                reward = self.rela_age[rly]
            else:
                age = -1
                self.render()
                reward = -100
            

            
        if next_mode:
            self.pkg = [1 for _ in range(self.rly_num)]
        else:
            self.pkg = [0 for _ in range(self.rly_num)]
        next_state = [self.h_,self.g_,self.data_normalize,self.eng_normalize,self.rela_age,self.pkg]
        return next_state, reward, age


    def test(self, mode, action, time, next_mode):
        act = int(action/self.rly_num)
        rly = int(action%self.rly_num)
        # print("Action: ", act, "Relay: ", rly)
        # cur_state = [self.h_,self.g_,self.snr1, self.snr2, self.data, self.eng, self.age[0].queue, self.age[1].queue,self.pkg, self.rela_age]
        # print("current state ", cur_state)
        # print("Time ", time)
        # input('Pause')

        # if mode!=1 and all(flag == 0 for flag in self.eng):   # It's EES state
        #     # for r in range(0,self.rly_num):
        #     #     self.eng[r] = self.eng[r]+ min(math.floor(self.h[r]*self.maxchannel*eg_eff*eng_ratio),self.eng_size)
        #     next_state = [self.h,self.g, self.data_normalize, self.eng_normalize, self.for_age_normalize, self.back_age_normalize]
        #     age = -1
        #     lost = 0
        #     self.render()
        #     return next_state, age, lost
        
        lost = 0
        if act == 0:
            if mode==1 and self.snr1[rly]<self.snr_th:
                lost += 1

            # elif mode==1 and recep==trans and self.eng[recep]!=0 and self.snr2[trans]>= self.snr_th:
            #     self.data[recep]+=1  # move agent
            #     for r in range(0,self.rly_num):  # transmit data to the reception reley, while the others charging
            #         if r != recep:
            #             self.eng[r]=min(math.floor(self.eng[r]+self.h[r]*self.maxchannel*eg_eff*eng_ratio),self.eng_size)
            #     self.age[recep].put(time)

            elif mode==1 and self.data[rly]!=self.data_size:
                self.data[rly]+=1  # move agent
                for r in range(0,self.rly_num):  # transmit data to the reception reley, while the others charging
                    if r != rly:
                        self.eng[r]=min(math.floor((self.eng[r]+self.h[r]*self.maxchannel*eg_eff*eng_ratio)*10)/10.0,self.eng_size)
                self.age[rly].put(time)


            elif mode==1 and self.data[rly]==self.data_size:
                for r in range(0,self.rly_num):  # transmit data to the reception reley, while the others charging
                    if r != rly:
                        self.eng[r]=min(math.floor((self.eng[r]+self.h[r]*self.maxchannel*eg_eff*eng_ratio)*10)/10.0,self.eng_size)
                self.age[rly].put(time)
                self.age[rly].get()
                lost += 1
            else :
                pass
            age = -1
        
        else:
            if self.data[rly]!=0 and self.eng[rly]>=1.0 and self.snr2[rly]>=self.snr_th:
                self.data[rly]-=1
                self.eng[rly]-=1.0
                age = self.age[rly].get()
            
                
            else:
                age = -1
        
        self.render()
        if next_mode:
            self.pkg = [1 for _ in range(self.rly_num)]
        else:
            self.pkg = [0 for _ in range(self.rly_num)]
        next_state = [self.h_,self.g_,self.data_normalize,self.eng_normalize,self.rela_age,self.pkg]
        return next_state, age, lost

    def test_two_phase(self, mode, action, time):
        recep = int(action/self.rly_num)
        trans = int(action%self.rly_num)
        # print("Recep: ", recep, "Trans: ", trans)
        # cur_state = [self.h_,self.g_,self.snr1, self.snr2, self.data, self.eng, self.age[0].queue, self.age[1].queue,self.age[2].queue]
        # print("current state ", cur_state)
        # print("Time ", time)
        # input('Pause')

        
        lost = 0

        if mode==1 and self.data[recep]!=self.data_size:
            self.data[recep]+=1  # move agent
            for r in range(0,self.rly_num):  # transmit data to the reception reley, while the others charging
                if r != recep:
                    self.eng[r]=min(math.floor((self.eng[r]+self.h[r]*self.maxchannel*eg_eff*eng_ratio)*10)/10.0,self.eng_size)
            self.age[recep].put(time)


        elif mode==1 and self.data[recep]==self.data_size:
            for r in range(0,self.rly_num):  # transmit data to the reception reley, while the others charging
                if r != recep:
                    self.eng[r]=min(math.floor((self.eng[r]+self.h[r]*self.maxchannel*eg_eff*eng_ratio)*10)/10.0,self.eng_size)
            self.age[recep].put(time)
            self.age[recep].get()
            lost += 1
        else :
            pass
        age = -1


        
        
        if self.data[trans]!=0 and self.eng[trans]>=1.0 and self.snr2[trans]>=self.snr_th:
            self.data[trans]-=1
            self.eng[trans]-=1.0
            age = self.age[trans].get()
            
                
        else:
            age = -1
        
        
        
        self.render()
        next_state = [self.h_,self.g_,self.data_normalize, self.eng_normalize]
        return next_state, age, lost

    def test_maxlink(self, mode, action, rly, time):
        # print("Recep: ", recep, "Trans: ", trans)
        # cur_state = [self.h_,self.g_,self.snr1, self.snr2, self.data, self.eng, self.age[0].queue, self.age[1].queue,self.age[2].queue]
        # print("current state ", cur_state)
        # print("Time ", time)
        # input('Pause')

        # if mode!=1 and all(flag == 0 for flag in self.eng):   # It's EES state
        #     # for r in range(0,self.rly_num):
        #     #     self.eng[r] = self.eng[r]+ min(math.floor(self.h[r]*self.maxchannel*eg_eff*eng_ratio),self.eng_size)
        #     next_state = [self.h,self.g, self.data_normalize, self.eng_normalize, self.for_age_normalize, self.back_age_normalize]
        #     age = -1
        #     lost = 0
        #     self.render()
        #     return next_state, age, lost
        
        lost = 0
        if action == 1:
            if mode==1 and self.data[rly]!=self.data_size:
                self.data[rly]+=1  # move agent
                for r in range(0,self.rly_num):  # transmit data to the reception reley, while the others charging
                    if r != rly:
                        self.eng[r]=min(math.floor((self.eng[r]+self.h[r]*self.maxchannel*eg_eff*eng_ratio)*10)/10.0,self.eng_size)
                self.age[rly].put(time)


            elif mode==1 and self.data[rly]==self.data_size:
                for r in range(0,self.rly_num):  # transmit data to the reception reley, while the others charging
                    if r != rly:
                        self.eng[r]=min(math.floor((self.eng[r]+self.h[r]*self.maxchannel*eg_eff*eng_ratio)*10)/10.0,self.eng_size)
                self.age[rly].put(time)
                self.age[rly].get()
                lost += 1
            else :
                pass
            age = -1
        
                
        elif action == 0:
            self.data[rly]-=1
            self.eng[rly]-=1.0
            age = self.age[rly].get()

        else:
            age = -1
            if mode == 1:
                lost += 1
            
                
        
        self.render()
        next_state = [self.h_,self.g_,self.data_normalize, self.eng_normalize]
        return next_state, age, lost

            
    def render(self):
        self.h = [self.random_channel[random.randrange(0, len(self.random_channel))] for _ in range(self.rly_num)]
        self.g = [self.random_channel[random.randrange(0, len(self.random_channel))] for _ in range(self.rly_num)]
        self.snr1 = [Ps*self.h[i]*self.maxchannel/N0 for i in range(self.rly_num)]
        self.snr2 = [Pr*self.g[i]*self.maxchannel/N0 for i in range(self.rly_num)]
        for r in range(0,self.rly_num):
            if self.snr1[r]>=self.snr_th:
                self.h_[r] = min(math.floor(self.h[r]*self.maxchannel*eg_eff*eng_ratio*10)/10.0,self.eng_size)
            else:
                self.h_[r] = -1
        self.g_ = [1 if s2>self.snr_th else 0 for s2 in self.snr2]
        for i in range(self.rly_num):
            if self.age[i].empty():
                self.for_age[i]=0
                self.back_age[i] = 0
            else:
                self.for_age[i]=list(self.age[i].queue)[0]
                self.back_age[i]=list(self.age[i].queue)[-1]
        
        self.data_normalize = [x / (self.data_size) for x in self.data]
        self.eng_normalize = [x / (self.eng_size) for x in self.eng]

        self.rela_age = [(self.age[r].queue[0]-self.current_age)/100 if not self.age[r].empty() else 0 for r in range(self.rly_num)]
                
                
            

               

    
