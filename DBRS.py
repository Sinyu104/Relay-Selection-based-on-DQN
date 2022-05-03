from pickle import NONE
import random

from sqlalchemy import null

RS = []
TS = []
RCS = []
TCS = []


def setRS(snr1, snr_th):
    RS.clear()
    for s in range(len(snr1)):
        if(snr1[s] > snr_th):
            RS.append(s)
        else:
            pass
    if(len(RS)==0):
        return False
    else:
        return True

def setTS(snr2, snr_th):
    TS.clear()
    for s in range(len(snr2)):
        if(snr2[s] > snr_th):
            TS.append(s)
        else:
            pass
    if(len(TS)==0):
        return False
    else:
        return True

def setRCS(data_bf):
    RCS.clear()
    minDatalen = 100
    
    for rly in RS:
        if(data_bf[rly] < minDatalen):
            RCS.clear()
            minDatalen = data_bf[rly]
            RCS.append(rly)
        elif (data_bf[rly]==minDatalen):
            RCS.append(rly)
        else:
            pass

def setTCS(data_bf):
    TCS.clear()
    maxDatalen = 0
    for rly in TS: 
        if(data_bf[rly]>maxDatalen):
            TCS.clear()
            TCS.append(rly)
            maxDatalen = data_bf[rly]
        elif (data_bf[rly]==maxDatalen):
            TCS.append(rly)
        else:
            pass

def DBRSReceive(eng_bf, data_bf, datasize):
    greateng = eng_bf[RCS[0]]
    candicate = []
    
    for rly in RCS:
        if(eng_bf[rly]>greateng):
            candicate.clear()
            candicate.append(rly)
            greateng = eng_bf[rly]
        elif(eng_bf[rly]==greateng):
            candicate.append(rly)
        else:
            pass
    

    if len(candicate)==1:
        recep_rly = candicate[0]
    else:
        recep_rly = random.choice(candicate)
    
    recep = recep_rly
    
    if(data_bf[recep_rly]==datasize-1):
        return -1
        
    else:
        return recep

def DBRSTransmit(eng_bf, data_bf):
    candicate = []
    greateng = eng_bf[TCS[0]]
    
    for rly in TCS:
        if(eng_bf[rly]>greateng):
            candicate.clear()
            candicate.append(rly)
            greateng = eng_bf[rly]
        elif(eng_bf[rly]==greateng):
            candicate.append(rly)
        else:
            pass

    if len(candicate)==1:
        trans_rly = candicate[0]
        
    else:
        trans_rly = random.choice(candicate)
    
    trans = trans_rly

    if(eng_bf[rly]<1 or data_bf[rly]<1):
        return -1
    else:
        return trans


def DBRS(rly_num, data_size, snr1, snr2, data_bf, eng_bf, snr_th=3):
    if(setRS(snr1, snr_th) and setTS(snr2, snr_th)): 
        setRCS(data_bf)  
        setTCS(data_bf) 
        recep = DBRSReceive(eng_bf, data_bf, data_size)
        trans = DBRSTransmit(eng_bf, data_bf)
        if(recep!=-1 and trans!=-1):
            return recep*rly_num+trans
        else:
            return -1
    else :
        return -1
                   