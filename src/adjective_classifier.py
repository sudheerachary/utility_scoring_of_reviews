import os
import numpy as np
from collections import defaultdict

def adj_predictor(text):
    wordlist=text.split(' ')
    count=[]
    filenames=os.listdir('./adjectives')
    for i in filenames:
        c=0
        ad_list=defaultdict(int)
        fp=open('./adjectives/'+i,'r').readlines()
        temp=fp[0].split(',')
        for word in temp:
            ad_list[word]=1
        for word in wordlist:
            if ad_list[word]==1:
                c+=1
        count.append(float(c)/len(ad_list.keys()))
    count=np.asarray(count)
    return np.argmax(count,axis=0)
        
