import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import matthews_corrcoef
from sklearn.feature_extraction.text import TfidfVectorizer
import re,random,os
import nltk
import sys

def shallow_syntactic_features(text):
	no_of_sentences=len(nltk.sent_tokenize(text))
	tokenized=nltk.word_tokenize(text)
	no_of_words=len(tokenized)
	pos=nltk.pos_tag(tokenized)
	NNP=0;CD=0;MD=0;UH=0;JJR=0;JJS=0;RBR=0;RBS=0;W=0
	for i in pos:
		if(i[1]=="NNP"):
			NNP+=1
		elif(i[1]=="CD"):
			CD+=1
		elif(i[1]=="MD"):
			MD+=1
		elif(i[1]=="UH"):
			UH+=1
		elif(i[1]=="JJR"):
			JJR+=1
		elif(i[1]=="JJS"):
			JJS+=1
		elif(i[1]=="RBR"):
			RBR+=1
		elif(i[1]=="RBS"):
			RBS+=1
		elif(i[1]=="WDT" or i[1]=="WP" or i[1]=="WP$" or i[1]=="WRB"):
			W+=1

	return np.asarray([NNP,CD,MD,UH,JJR,JJS,RBR,RBS,W,no_of_words,no_of_sentences])



#dynamic
fdyn=open("dynamic.txt","r")
dyn={}
for line in fdyn:
	line=line.strip("\n")
	dyn[line]=1

#polar-neg
fplrneg=open("polar-neg.txt","r")
plr_neg={}
for line in fplrneg:
	line=line.strip("\n")
	plr_neg[line]=1

#polar-pos
fplrpos=open("polar-pos.txt","r")
plr_pos={}
for line in fplrpos:
	line=line.strip("\n")
	plr_pos[line]=1

#gradable-pos
fgradpos=open("gradable-pos.txt","r")
grad_pos={}
for line in fgradpos:
	line=line.strip("\n")
	grad_pos[line]=1

#grad-neg
fgradneg=open("gradable-neg.txt","r")
grad_neg={}
for line in fgradneg:
	line=line.strip("\n")
	grad_neg[line]=1

#strng-adj-neg
fstrngadgneg=open("strongsubj-adj-neg.txt")
strng_adj_neg={}
for line in fstrngadgneg:
	line=line.strip("\n")
	strng_adj_neg[line]=1

#strng-adj-pos
fstrngadgpos=open("strongsubj-adj-pos.txt")
strng_adj_pos={}
for line in fstrngadgpos:
	line=line.strip("\n")
	strng_adj_pos[line]=1

#strng-noun-neg
fstrngnounneg=open("strongsubj-noun-neg.txt")
strng_noun_neg={}
for line in fstrngnounneg:
	line=line.strip("\n")
	strng_noun_neg[line]=1

#strng-noun-pos
fstrngnounpos=open("strongsubj-noun-pos.txt")
strng_noun_pos={}
for line in fstrngnounpos:
	line=line.strip("\n")
	strng_noun_pos[line]=1

#wk-adj-neg
fwkadgneg=open("weaksubj-adj-neg.txt")
wk_adj_neg={}
for line in fwkadgneg:
	line=line.strip("\n")
	wk_adj_neg[line]=1

#wk-adj-pos
fwkadgpos=open("weaksubj-adj-pos.txt")
wk_adj_pos={}
for line in fwkadgpos:
	line=line.strip("\n")
	wk_adj_pos[line]=1

#wk-noun-neg
fwknounneg=open("weaksubj-noun-neg.txt")
wk_noun_neg={}
for line in fwknounneg:
	line=line.strip("\n")
	wk_noun_neg[line]=1

#wk-adj-pos
fwknounpos=open("weaksubj-noun-pos.txt")
wk_noun_pos={}
for line in fwknounpos:
	line=line.strip("\n")
	wk_noun_pos[line]=1



def feature(text):
	tokenized=nltk.word_tokenize(text)
	d=0;p_n=0;p_p=0;g_n=0;g_p=0;s_a_n=0;s_a_p=0;s_n_n=0;s_n_p=0;w_a_n=0;w_a_p=0;w_n_n=0;w_n_p=0
	for i in tokenized:
		if i in dyn:
			d+=1
		if i in plr_neg:
			p_n+=1
		if i in plr_pos:
			p_p+=1
		if i in grad_neg:
			g_n+=1
		if i in grad_pos:
			g_p+=1
		if i in strng_adj_neg:
			s_a_n+=1
		if i in strng_adj_pos:
			s_a_p+=1
		if i in strng_noun_neg:
			s_n_n+=1
		if i in strng_noun_pos:
			s_n_p+=1
		if i in wk_adj_neg:
			w_a_n+=1
		if i in wk_adj_pos:
			w_a_p+=1
		if i in wk_noun_neg:
			w_n_n+=1
		if i in wk_noun_pos:
			w_n_p+=1
	return np.asarray([d, p_n, p_p, g_n, g_p, s_a_n, s_a_p, s_n_n, s_n_p, w_a_n, w_a_p, w_n_n, w_n_p])

# fp=open(sys.argv[1],"r")
# filename=sys.argv[1].split("/")
# filename=re.sub('.csv','',filename[len(filename)-1])

fout=open("./outputs/"+'all_combined_3k'+".txt","w")
data=[]
datalbl=[]
full_data=[]
totalcount=0
reviews=0
for d in os.listdir('DATASETS/final/'):
	print d
	fp=open('DATASETS/final/'+d,'r')
	f=csv.reader(fp)
	daata=[]
	for line in f:
		totalcount+=1
		try:
			if(int(line[1])>=10):
				temp=line[2].split()
				for j in range(0,len(temp)):
					if("http" in temp[j]):
						temp[j]=''
					else:
						temp[j]=temp[j].lower()
				line[2]=' '.join(temp)
				pure_line=re.sub('<.*?>','',line[2])
				daata.append((pure_line,int(line[0])*1.0/int(line[1])))
			# data.append(pure_line)
			# datalbl.append(int(line[0])*1.0/int(line[1]))
			# print line[5]
				reviews+=1
		except:
			pass
	random.shuffle(daata)
	full_data.extend(daata[:3000])
random.shuffle(full_data)
for i in full_data:
	data.append(i[0])
	datalbl.append(i[1])
print "total lines in file",totalcount
print "selected reviews",reviews
fout.write("total lines in file :"+str(totalcount)+"\n")
fout.write("selected reviews :"+str(reviews)+"\n\n\n")
X_trainpart, X_testpart, y_train, y_test = train_test_split(data, datalbl, test_size=0.1, random_state=0)
print "SPLIT done"

X_train1=[]
X_test1=[]
X_train2=[]
X_test2=[]
X_train3=[]
X_test3=[]

for i in range(len(X_trainpart)):
	x=feature(X_trainpart[i].decode("utf8"))
	y=shallow_syntactic_features(X_trainpart[i].decode("utf8"))
	X_train1.append(x)
	X_train2.append(y)
	X_train3.append(np.concatenate([x,y]))


for i in range(len(X_testpart)):
	x=feature(X_testpart[i].decode("utf8"))
	y=shallow_syntactic_features(X_testpart[i].decode("utf8"))
	X_test1.append(x)
	X_test2.append(y)
	X_test3.append(np.concatenate([x,y]))

print "--------Shallow Synatactic Features-------------"
fout.write("-----Shallow Synatactic Features--------"+"\n\n")
print "****SLR****"
fout.write("****SLR****"+"\n")
slr=linear_model.LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=1)
slr.fit(X_train1,y_train)
print "model trained"

prdctslr=slr.predict(X_test1)
for i in range(len(prdctslr)):
    prdctslr[i]=min(1.0,prdctslr[i])
print "model predicted"

prdctslrsub=prdctslr-np.mean(prdctslr)
y_testsub=y_test-np.mean(y_test)
temp1=np.sum(np.multiply(prdctslrsub,y_testsub))
tempn1=temp1*temp1
tempd1=np.sum(np.multiply(prdctslrsub,prdctslrsub))
tempd2=np.sum(np.multiply(y_testsub,y_testsub))
slr_scc=tempn1/(tempd1*tempd2)
print "Squared Correlation Coefficient",slr_scc
fout.write("Squared Correlation Coefficient :"+str(slr_scc)+"\n")

slr_mse=np.dot(np.subtract(prdctslr,y_test),np.subtract(prdctslr,y_test))/len(y_test)
print "Mean Squared Error",slr_mse
fout.write("Mean Squared Error :"+str(slr_mse)+"\n\n")

print "*****SVR*****"
fout.write("****SVR****"+"\n")

svr=svm.SVR()
svr.fit(X_train1,y_train)
print "model trained"

prdctsvr=svr.predict(X_test1)
for i in range(len(prdctsvr)):
    prdctsvr[i]=min(1.0,prdctsvr[i])
print "model predicted"

prdctsvrsub=prdctsvr-np.mean(prdctsvr)
y_testsub=y_test-np.mean(y_test)
temp1=np.sum(np.multiply(prdctsvrsub,y_testsub))
tempn1=temp1*temp1
tempd1=np.sum(np.multiply(prdctsvrsub,prdctsvrsub))
tempd2=np.sum(np.multiply(y_testsub,y_testsub))
svr_scc=tempn1/(tempd1*tempd2)
print "Squared Correlation Coefficient",svr_scc
fout.write("Squared Correlation Coefficient :"+str(svr_scc)+"\n")


svr_mse=np.dot(np.subtract(prdctsvr,y_test),np.subtract(prdctsvr,y_test))/len(y_test)
print "Mean Squared Error",svr_mse
fout.write("Mean Squared Error :"+str(svr_mse)+"\n\n\n")


print "----------Lexical Subjective Clues-------------------"
fout.write("----Lexical Subjective Clues------"+"\n\n")
print "****SLR****"
fout.write("****SLR****"+"\n")

slr=linear_model.LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=1)
slr.fit(X_train2,y_train)
print "model trained"

prdctslr=slr.predict(X_test2)
for i in range(len(prdctslr)):
    prdctslr[i]=min(1.0,prdctslr[i])
print "model predicted"

prdctslrsub=prdctslr-np.mean(prdctslr)
y_testsub=y_test-np.mean(y_test)
temp1=np.sum(np.multiply(prdctslrsub,y_testsub))
tempn1=temp1*temp1
tempd1=np.sum(np.multiply(prdctslrsub,prdctslrsub))
tempd2=np.sum(np.multiply(y_testsub,y_testsub))
slr_scc=tempn1/(tempd1*tempd2)
print "Squared Correlation Coefficient",slr_scc
fout.write("Squared Correlation Coefficient :"+str(slr_scc)+"\n")


slr_mse=np.dot(np.subtract(prdctslr,y_test),np.subtract(prdctslr,y_test))/len(y_test)
print "Mean Squared Error",slr_mse
fout.write("Mean Squared Error :"+str(slr_mse)+"\n\n")


print "*****SVR*****"
fout.write("****SVR****"+"\n")

svr=svm.SVR()
svr.fit(X_train2,y_train)
print "model trained"

prdctsvr=svr.predict(X_test2)
for i in range(len(prdctsvr)):
    prdctsvr[i]=min(1.0,prdctsvr[i])
print "model predicted"

prdctsvrsub=prdctsvr-np.mean(prdctsvr)
y_testsub=y_test-np.mean(y_test)
temp1=np.sum(np.multiply(prdctsvrsub,y_testsub))
tempn1=temp1*temp1
tempd1=np.sum(np.multiply(prdctsvrsub,prdctsvrsub))
tempd2=np.sum(np.multiply(y_testsub,y_testsub))
svr_scc=tempn1/(tempd1*tempd2)
print "Squared Correlation Coefficient",svr_scc
fout.write("Squared Correlation Coefficient :"+str(svr_scc)+"\n")



svr_mse=np.dot(np.subtract(prdctsvr,y_test),np.subtract(prdctsvr,y_test))/len(y_test)
print "Mean Squared Error",svr_mse
fout.write("Mean Squared Error :"+str(svr_mse)+"\n\n\n")


print "----------Shallow Synatactic Features+Lexical Subjective Clues----------"
fout.write("-----Shallow Synatactic Features+Lexical Subjective Clues------"+"\n\n")

print "****SLR****"
fout.write("****SLR****"+"\n")

slr=linear_model.LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=1)
slr.fit(X_train3,y_train)
print "model trained"

prdctslr=slr.predict(X_test3)
for i in range(len(prdctslr)):
    prdctslr[i]=min(1.0,prdctslr[i])
print "model predicted"

prdctslrsub=prdctslr-np.mean(prdctslr)
y_testsub=y_test-np.mean(y_test)
temp1=np.sum(np.multiply(prdctslrsub,y_testsub))
tempn1=temp1*temp1
tempd1=np.sum(np.multiply(prdctslrsub,prdctslrsub))
tempd2=np.sum(np.multiply(y_testsub,y_testsub))
slr_scc=tempn1/(tempd1*tempd2)
print "Squared Correlation Coefficient",slr_scc
fout.write("Squared Correlation Coefficient :"+str(slr_scc)+"\n")


slr_mse=np.dot(np.subtract(prdctslr,y_test),np.subtract(prdctslr,y_test))/len(y_test)
print "Mean Squared Error",slr_mse
fout.write("Mean Squared Error :"+str(slr_mse)+"\n\n")



print "*****SVR*****"
fout.write("****SVR****"+"\n")

svr=svm.SVR()
svr.fit(X_train3,y_train)
print "model trained"

prdctsvr=svr.predict(X_test3)
for i in range(len(prdctsvr)):
    prdctsvr[i]=min(1.0,prdctsvr[i])
print "model predicted"

prdctsvrsub=prdctsvr-np.mean(prdctsvr)
y_testsub=y_test-np.mean(y_test)
temp1=np.sum(np.multiply(prdctsvrsub,y_testsub))
tempn1=temp1*temp1
tempd1=np.sum(np.multiply(prdctsvrsub,prdctsvrsub))
tempd2=np.sum(np.multiply(y_testsub,y_testsub))
svr_scc=tempn1/(tempd1*tempd2)
print "Squared Correlation Coefficient",svr_scc
fout.write("Squared Correlation Coefficient :"+str(svr_scc)+"\n")



svr_mse=np.dot(np.subtract(prdctsvr,y_test),np.subtract(prdctsvr,y_test))/len(y_test)
print "Mean Squared Error",svr_mse
fout.write("Mean Squared Error :"+str(svr_mse)+"\n")

fout.close()

