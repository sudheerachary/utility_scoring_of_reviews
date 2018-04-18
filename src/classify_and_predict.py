import os,sys,time,csv,random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.preprocessing import text
import keras
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import matthews_corrcoef
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
import nltk
from keras.models import model_from_json
import pickle

filemap = {}

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

def one_hot_vec(train_data,vocablength):
    train_data = np.asarray(train_data)
    X_train=[]
    print '--- one hot encoding ---'
    for i in train_data:
        temp = text.one_hot(' '.join(i),vocablength)
        X_train.append(temp)
    X_train = sequence.pad_sequences(X_train, maxlen=500)
    return np.asarray(X_train)

def get_data():
	filepos = 0
	ttl_data=[]
	temp_data_nn=[]
	for filename in os.listdir("./DATASETS/final"):
		filemap[filepos] = re.sub('.csv', '', filename)
		filename = "./DATASETS/final/" + filename
		print filename		
		fp = open(filename, 'r')
		f = csv.reader(fp)
		data = []
		reviews = 0
		for line in f:
			if int(line[1]) >= 10:
				if reviews>1100:
					break
				temp = line[2].split()
				for j in range(0, len(temp)):
					if "http" in temp[j]:
						temp[j] = ''
					else:
						temp[j] = temp[j].lower()
				line[2] = ' '.join(temp)
				pure_line = re.sub('<.*?>', '', line[2])
				#data.append((pure_line,float(int(line[0])/int(line[1]))))
				data.append((pure_line,filepos,float(line[0])/int(line[1])))	
				#print line[5]
				reviews += 1
		random.shuffle(data)
		data = data[:1000]
		fdata = []
		ttl_data.extend(data)
		for pure_line in data:
			temp = [word for word in pure_line[0].split() if word not in (stopwords.words('english'))]
			fdata.append(temp)
		temp_data_nn.extend(fdata)
		filepos += 1

	train_data = [i[0] for i in ttl_data]
	labels_crct = [i[1] for i in ttl_data]
	u_scores=[i[2] for i in ttl_data]
	X_train_prdct=[]
	for i in train_data:
		x=feature(i.decode("utf8"))
		y=shallow_syntactic_features(i.decode("utf8"))
		X_train_prdct.append(np.concatenate([x,y]))

	X_train_nn=one_hot_vec(temp_data_nn,1135971)
	
	return np.asarray(X_train_prdct), labels_crct , np.asarray(X_train_nn),u_scores

def nn_predict(data,labels):
	json_file = open('./classifiers/nn.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("./classifiers/nn.h5")
	
	y_pred=loaded_model.predict(data)
	pred_labels = np.argmax(y_pred, axis=1)
	positives = 0;
	for i in range(y_pred.shape[0]):
		#print "prediction: "+str(pred_labels[i])+" truth: "+str(labels[i])
		if pred_labels[i] == labels[i]:
			positives += 1
	print '--- Accuracy ---: ', float(positives)/float(y_pred.shape[0])
	return pred_labels

def nn_predict_scores(X_train_prdct,labels,slr_models,svr_models,u_scores):
	count=0
	slr_predict=[]
	svr_predict=[]
	for i in range(len(labels)):
		#print "review number:",count,"label: ",labels[i]," slr_prediction ",slr_models[filemap[labels[i]]].predict([X_train_prdct[i]])
		#print "review number:",count,"label: ",labels[i]," svr_prediction ",svr_models[filemap[labels[i]]].predict([X_train_prdct[i]])
		slr_predict.extend(slr_models[filemap[labels[i]]].predict([X_train_prdct[i]]))
		svr_predict.extend(svr_models[filemap[labels[i]]].predict([X_train_prdct[i]]))		
		count+=1
	print count
	foutslr=open("classifiedoutputs/slr_classified.txt",'w')
	for i in range(len(slr_predict)):
		if(slr_predict[i]<0):
			slr_predict[i]=0
		slr_predict[i]=min(1.0,slr_predict[i])
		foutslr.write(str(slr_predict[i])+","+str(u_scores[i])+"\n")
	foutsvr=open("classifiedoutputs/svr_classified.txt",'w')
	for i in range(len(svr_predict)):
		if(svr_predict[i]<0):
			svr_predict[i]=0
		svr_predict[i]=min(1.0,svr_predict[i])
		foutsvr.write(str(svr_predict[i])+","+str(u_scores[i])+"\n")
	foutslr.close()
	foutsvr.close()
	return np.asarray(slr_predict),np.asarray(svr_predict)

if __name__ == '__main__':
	svr_models={}
	slr_models={}
	for filename in os.listdir('models'):
		if('svr' in filename):
        		svr_models[re.sub('svr_', '', filename)]=pickle.load(open('./models/'+filename, 'rb'))
		if('slr' in filename):
        		slr_models[re.sub('slr_', '', filename)]=pickle.load(open('./models/'+filename, 'rb'))
	print len(slr_models),len(svr_models)	
	X_train_prdct,labels_crct,X_train_nn,u_scores=get_data()
	#print filemap
	print "shapes",X_train_prdct.shape,X_train_nn.shape
	labels_nn=nn_predict(X_train_nn,labels_crct)
	#print labels_nn


	nn_slr,nn_svr=nn_predict_scores(X_train_prdct,labels_nn,slr_models,svr_models,u_scores)
	print "slr_shape",nn_slr.shape
	print "svr_shape",nn_svr.shape
	



	fout=open("classifiedoutputs/Errors_classified.txt","w")
	print "----svr----"
	fout.write("----svr----"+"\n")
	prdctsvrsub=nn_svr-np.mean(nn_svr)
	y_testsub=u_scores-np.mean(u_scores)
	temp1=np.sum(np.multiply(prdctsvrsub,y_testsub))
	tempn1=temp1*temp1
	tempd1=np.sum(np.multiply(prdctsvrsub,prdctsvrsub))
	tempd2=np.sum(np.multiply(y_testsub,y_testsub))
	svr_scc=tempn1/(tempd1*tempd2)
	print "Squared Correlation Coefficient",svr_scc
	fout.write("Squared Correlation Coefficient"+str(svr_scc)+"\n")

	svr_mse=np.dot(np.subtract(nn_svr,u_scores),np.subtract(nn_svr,u_scores))/len(u_scores)
	print "Mean Squared Error",svr_mse
	fout.write("Mean Squared Error"+str(svr_mse)+"\n")

	print "---slr---"
	fout.write("---slr---"+"\n")
	prdctsvrsub=nn_slr-np.mean(nn_slr)
	y_testsub=u_scores-np.mean(u_scores)
	temp1=np.sum(np.multiply(prdctsvrsub,y_testsub))
	tempn1=temp1*temp1
	tempd1=np.sum(np.multiply(prdctsvrsub,prdctsvrsub))
	tempd2=np.sum(np.multiply(y_testsub,y_testsub))
	slr_scc=tempn1/(tempd1*tempd2)
	print "Squared Correlation Coefficient",slr_scc
	fout.write("Squared Correlation Coefficient"+str(slr_scc)+"\n")

	slr_mse=np.dot(np.subtract(nn_slr,u_scores),np.subtract(nn_slr,u_scores))/len(u_scores)
	print "Mean Squared Error",slr_mse
	fout.write("Mean Squared Error"+str(slr_mse)+"\n")
	fout.close()

