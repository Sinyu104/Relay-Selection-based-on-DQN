from cmath import inf
import numpy as np
from numpy import Inf
import random
#from send2trash import TrashPermissionError

def SAR_LAT(rly_num, snr1, snr2, age, data_bf, eng_bf, snr_th=3):
    recep_candicate = []
    trans_candicate = []    

    for i in range(rly_num):
        if data_bf[i]==0:
            recep_candicate.append(i)
    if recep_candicate:
        if len(recep_candicate)==1:
            recep = recep_candicate[0]
        else:
            recep = random.choice(recep_candicate)
    else:
        max_age = Inf
        for i in range(rly_num):
            if list(age[i].queue)[0] <max_age:
                recep = i
                max_age = list(age[i].queue)[0]
            else:
                pass
    
    if snr1[recep]<snr_th:
        recep = -1

    for i in range(rly_num):
        if data_bf[i]==1:
            trans_candicate.append(i)
    if trans_candicate:
        if len(trans_candicate)==1:
            trans = trans_candicate[0]
        else:
             max_age = Inf
             for i in trans_candicate:
                if list(age[i].queue)[0] <max_age:
                    trans = i
                    max_age = list(age[i].queue)[0]
    else:
        return -1

    # if snr2[trans]<snr_th or data_bf[trans]==0 or eng_bf[trans]<1.0:
    #     trans = -1


    # if recep == -1 and trans == -1:
    #     return -1
    # elif recep==-1 and trans !=-1:
    #     recep = random.randrange(rly_num)
    #     return recep*rly_num+trans
    # elif recep!=-1 and trans ==-1:
    #     trans = random.randrange(rly_num)
    #     return recep*rly_num+trans
    if snr1[recep]<snr_th or snr2[trans]<snr_th or data_bf[trans]==0 or eng_bf[trans]<1.0:
        return -1
    else:
        return recep*rly_num+trans
