import os,sys
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import matthews_corrcoef
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from collections import defaultdict
fp=open(sys.argv[1],'r')
f=csv.reader(fp)
data=[]
datalbl=[]
adjec=defaultdict(int)
count=0
for line in f:
    try:
        if(int(line[1])>=10):
            count+=1
            if(count>30000):
                break
            temp=line[2].split()
            for j in range(0,len(temp)):
                if("http" in temp[j]):
                    temp[j]=''
                else:
                    temp[j]=temp[j].lower()
            line[2]=' '.join(temp)
            pure_line=re.sub('<.*?>','',line[2])
            data.append(pure_line)
    except:
        pass
for i in data:
    i=i.decode("utf8")
    tokenized=nltk.word_tokenize(i)
    pos=nltk.pos_tag(tokenized)
    for p in pos:
        if(p[1]=='JJ'):
            adjec[p[0]]=1
filename=sys.argv[1].split('/')[-1]
f=open(filename+'_adj_list','w')
m=adjec.keys()
print len(m)
st=','.join(m)
f.write(st.encode('utf-8'))
