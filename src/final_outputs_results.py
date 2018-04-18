import os,time,re,sys
import nltk
import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
from sklearn.mixture import GMM
import matplotlib as mpl
import csv
from keras.preprocessing import text
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import matthews_corrcoef
from sklearn.feature_extraction.text import TfidfVectorizer

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


def regressor(X_ftrain,train_utilities,X_ftest,test_utilities,fout,filename):
	fout.write("------SLR------"+"\n")
	slr=linear_model.LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=1)
	slr.fit(X_ftrain,train_utilities)
	print "model trained"
	prdctslr=slr.predict(X_ftest)
	for i in range(len(prdctslr)):
		if(prdctslr[i]<0):
			prdctslr[i]=0
		prdctslr[i]=min(1.0,prdctslr[i])
	# prdctslr = (prdctslr-np.min(prdctslr))*1.0/(np.max(prdctslr))
	print "model predicted"	
	plt.plot(prdctslr)
	plt.plot(test_utilities)
	plt.legend(['predictions', 'ground truth'], loc='upper left')
	plt.savefig('results/'+filename+'-slr.png')
	plt.clf()

	prdctslrsub=prdctslr-np.mean(prdctslr)
	y_testsub=test_utilities-np.mean(test_utilities)
	temp1=np.sum(np.multiply(prdctslrsub,y_testsub))
	tempn1=temp1*temp1
	tempd1=np.sum(np.multiply(prdctslrsub,prdctslrsub))
	tempd2=np.sum(np.multiply(y_testsub,y_testsub))
	slr_scc=tempn1/(tempd1*tempd2)
	print "Squared Correlation Coefficient",slr_scc
	fout.write("Squared Correlation Coefficient :"+str(slr_scc)+"\n")

	slr_mse=np.dot(np.subtract(prdctslr,test_utilities),np.subtract(prdctslr,test_utilities))/len(test_utilities)
	print "Mean Squared Error",slr_mse
	fout.write("Mean Squared Error :"+str(slr_mse)+"\n\n")

	print "*****SVR*****"
	fout.write("------SVR------"+"\n")

	svr=svm.SVR()
	svr.fit(X_ftrain,train_utilities)
	print "model trained"

	prdctsvr=svr.predict(X_ftest)
	for i in range(len(prdctsvr)):
		if(prdctsvr[i]<0):
			prdctsvr[i]=0
		prdctsvr[i]=min(1.0,prdctsvr[i])
	# prdctsvr = (prdctsvr-np.min(prdctsvr))*1.0/(np.max(prdctsvr))	
	print "model predicted"
	plt.plot(prdctsvr)
	plt.plot(test_utilities)
	plt.legend(['predictions', 'ground truth'], loc='upper left')
	plt.savefig('results/'+filename+'-svr.png')
	plt.clf()

	prdctsvrsub=prdctsvr-np.mean(prdctsvr)
	y_testsub=test_utilities-np.mean(test_utilities)
	temp1=np.sum(np.multiply(prdctsvrsub,y_testsub))
	tempn1=temp1*temp1
	tempd1=np.sum(np.multiply(prdctsvrsub,prdctsvrsub))
	tempd2=np.sum(np.multiply(y_testsub,y_testsub))
	svr_scc=tempn1/(tempd1*tempd2)
	print "Squared Correlation Coefficient",svr_scc
	fout.write("Squared Correlation Coefficient :"+str(svr_scc)+"\n")

	svr_mse=np.dot(np.subtract(prdctsvr,test_utilities),np.subtract(prdctsvr,test_utilities))/len(test_utilities)
	print "Mean Squared Error",svr_mse
	fout.write("Mean Squared Error :"+str(svr_mse)+"\n\n\n")




fp=open(sys.argv[1],"r")
temp=sys.argv[1].split("/")
temp=re.sub(".csv",'',temp[3])
filename = temp
fout=open("finaloutputs/"+temp+".txt","w")
f=csv.reader(fp)
data=[]
count=0
for line in f:
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
			#pure_line=re.sub('/','',pure_line)
			# pure_line=re.sub('+','',pure_line)
			#print pure_line
			tokens=nltk.word_tokenize(pure_line)
			pos_tags=nltk.pos_tag(tokens)
			pure_line1=[]
			for i in pos_tags:
				if(i[1]=="NN"):
					if(i[0]=='/' or i[0]=='+'):
							continue
					pure_line1.append(i[0])
				else:
					pure_line1.append(';;_')
			# if(len(pure_line1)==0):
						# continue
			data.append((pure_line1,int(line[0])*1.0/int(line[1])))
			count+=1
			if(count>10000):
				break
	except Exception as e:
		# print(e)
		pass

print len(data)
# print data
L=len(data)
data=np.asarray(data)
train_data=data[0:int(0.9*L),0]
train_utilities=data[0:int(0.9*L),1]
test_data=data[int(0.9*L):,0]
test_utilities=data[int(0.9*L):,1]
# train_data=data[0:3000]
train_data=np.asarray(train_data)
test_data=np.asarray(test_data)

vocabulory=np.unique(np.hstack(data))
# print vocabulory
print '---length---: ', len(vocabulory)
X_train=[];X_test=[];
length=0
for i in train_data:
	temp = text.one_hot(' '.join(i),len(vocabulory))
	length = max(length,len(temp))
	X_train.append(np.asarray(temp))
for i in test_data:
	temp = text.one_hot(' '.join(i),len(vocabulory))
	length = max(length,len(temp))
	X_test.append(temp)
X_train = sequence.pad_sequences(X_train, maxlen=length)
X_test = sequence.pad_sequences(X_test, maxlen=length)
classifier=GMM(n_components=20,covariance_type='full')
classifier.fit(X_train)
# y_gmm=classifier.predict(X_train)
# x=[i for i in range(0,len(y_gmm))]
# print y_gmm
# plt.scatter(x,y_gmm)
# plt.show()
means=np.round(classifier.means_, 2)
# print means[0]
# print X_train[0]
ls=len(X_train)
X_ftrain=[]
X_ftest=[]
# d=np.zeros((len(X_train),len(means))
trainadj=[]
trainsyntac=[]
traindist=[]
testadj=[]
testsyntac=[]
testdist=[]
print "generating vectors"
for i in range(len(X_train)):
	# dists=[]
	dist=99999999999999
	adj=feature(' '.join(train_data[i]))
	trainadj.append(adj)
	syntac=shallow_syntactic_features(' '.join(train_data[i]))
	trainsyntac.append(syntac)
	for j in range(0,len(means)):
		dist = min(dist,np.linalg.norm(X_train[i]-means[j]))
	traindist.append([dist])
	#dists.append(dist)
	#dists=np.asarray(dist)
	X_ftrain.append(np.concatenate((np.concatenate((adj,syntac)),np.asarray([dist]))))

for i in range(len(X_test)):
	dist=99999999999999
	adj=feature(' '.join(test_data[i]))
	testadj.append(adj)
	syntac=shallow_syntactic_features(' '.join(test_data[i]))
	testsyntac.append(syntac)
	for j in range(0,len(means)):
		dist = min(dist,np.linalg.norm(X_test[i]-means[j]))
		# dists.append(dist)
	testdist.append([dist])
	X_ftest.append(np.concatenate((np.concatenate((adj,syntac)),np.asarray([dist]))))

print len(trainadj)
print len(trainsyntac)
print len(traindist)
print traindist
testdist=np.asarray(testdist)
traindist=np.asarray(traindist)
# print (np.concatenate((trainadj,trainsyntac),axis=1)).shape
# np.concatenate((trainadj,trainsyntac),axis=0)
fout.write("--------------LEXICAL SUBJECTIVE CLUES--------------\n")
print "adjectives"
regressor(trainadj,train_utilities,testadj,test_utilities,fout,filename)

fout.write("------------SHALLOW SYNTACTIC FEATURES--------------\n")
print "syntacs"
regressor(trainsyntac,train_utilities,testsyntac,test_utilities,fout,filename)

fout.write("-------------CLUSTERING------------------------\n")
print "distances"
regressor(traindist,train_utilities,testdist,test_utilities,fout,filename)

fout.write("--------------LEXICAL SUBJECTIVE CLUES + SHALLOW SYNTACTIC FEATURES--------------\n")
print"adj+syntac"
regressor(np.concatenate((trainadj,trainsyntac),axis=1),train_utilities,np.concatenate((testadj,testsyntac),axis=1),test_utilities,fout,filename)

fout.write("--------------LEXICAL SUBJECTIVE CLUES + SHALLOW SYNTACTIC FEATURES + CLUSTERING--------------\n")
print "all togethert"
regressor(X_ftrain,train_utilities,X_ftest,test_utilities,fout,filename)
