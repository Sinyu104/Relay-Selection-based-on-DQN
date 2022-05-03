import random

def SAR_LAT(rly_num, data_size, data_bf, eng_bf, cur_age):
    smallest_age = float('inf')
    biggest_age = float('-inf')
    recep_candicate = []
    trans_candicate = []
    for i in range(rly_num):
        if data_bf[i] == data_size-1:
            continue
        if cur_age[i] <smallest_age:
            smallest_age = cur_age[i]
            recep_candicate.clear()
            recep_candicate.append(i)
        elif cur_age[i] == smallest_age:
            recep_candicate.append(i)
        else:
            pass
    for i in range(rly_num):
        if data_bf[i] == 0 or eng_bf[i] == 0:
            continue
        if cur_age[i] >biggest_age:
            biggest_age = cur_age[i]
            trans_candicate.clear()
            trans_candicate.append(i)
        elif cur_age[i] == biggest_age:
            trans_candicate.append(i)
        else:
            pass
    
    action = random.choice(recep_candicate)*rly_num+random.choice(trans_candicate) if len(recep_candicate)!=0 and len(trans_candicate)!=0 else -1
    return action