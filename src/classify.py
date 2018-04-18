from keras.models import Sequential
from keras.layers import Dense
from nltk.corpus import stopwords
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing import text
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn import linear_model
from scipy import sparse
from sklearn.metrics import matthews_corrcoef
from sklearn.feature_extraction.text import TfidfVectorizer
import os, sys, time, random, csv, re
import numpy as np
import pickle
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

def get_data():
	filemap = {}
	filepos = 0
	train_data = []
	test_data = []
	for filename in os.listdir("./DATASETS/final"):
		filename = "./DATASETS/final/" + filename
		print filename
		filemap[filename] = filepos
		fp = open(filename, 'r')
		f = csv.reader(fp)
		data = []
		reviews = 0
		for line in f:
			if int(line[1]) >= 10:
				temp = line[2].split()
				for j in range(0, len(temp)):
					if "http" in temp[j]:
						temp[j] = ''
					else:
						temp[j] = temp[j].lower()
				line[2] = ' '.join(temp)
				pure_line = re.sub('<.*?>', '', line[2])
				#data.append((pure_line,float(int(line[0])/int(line[1]))))
				data.append((pure_line,filepos))
				#print line[5]
				reviews += 1
		random.shuffle(data)
		data = data[:10000]
		fdata = []
		for pure_line in data:
			temp = [word for word in pure_line[0].split() if word not in (stopwords.words('english'))]
			fdata.append((temp, pure_line[1]))
		train_data.extend(fdata[0:int(0.9*len(data))])
		test_data.extend(fdata[int(0.9*len(data)):])
		filepos += 1

	train_data1 = [i[0] for i in train_data]
	train_labels = [i[1] for i in train_data]
	test_data1 = [i[0] for i in test_data]
	test_labels = [i[1] for i in test_data]
	
	train_data1 = np.asarray(train_data1)
 	test_data1 = np.asarray(test_data1)
	X = np.concatenate((train_data1, test_data1), axis=0)
	vocabulory = np.unique(np.hstack(X))
	print '---length---: ', len(vocabulory)
	X_train=[];X_test=[];length = 0
	print '--- one hot encoding ---'
	for i in train_data1:
		temp = text.one_hot(' '.join(i),len(vocabulory))
		length = max(length,len(temp))
		X_train.append(temp)		
	for i in test_data1:
		temp = text.one_hot(' '.join(i),len(vocabulory))
		length = max(length,len(temp))
		X_test.append(temp)

	X_train = sequence.pad_sequences(X_train, maxlen=500)
	X_test = sequence.pad_sequences(X_test, maxlen=500)
	np.save('train_X.npy', train_data1)
	np.save('test_X.npy', test_data1)
	with open('train_Y.txt', 'wb') as f:
		pickle.dump(train_labels, f)
	with open('test_Y.txt', 'wb') as f:
		pickle.dump(test_labels, f)
	return np.asarray(X_train), train_labels, np.asarray(X_test), test_labels

def one_hot_vec(train_data1,test_data1):
	train_data1 = np.asarray(train_data1)
 	test_data1 = np.asarray(test_data1)
	X = np.concatenate((train_data1, test_data1), axis=0)
	vocabulory = np.unique(np.hstack(X))
	print '---length---: ', len(vocabulory)
	X_train=[];X_test=[];length = 0
	print '--- one hot encoding ---'
	for i in train_data1:
		temp = text.one_hot(' '.join(i),len(vocabulory))
		length = max(length,len(temp))
		X_train.append(temp)		
	for i in test_data1:
		temp = text.one_hot(' '.join(i),len(vocabulory))
		length = max(length,len(temp))
		X_test.append(temp)

	X_train = sequence.pad_sequences(X_train, maxlen=500)
	X_test = sequence.pad_sequences(X_test, maxlen=500)
	return np.asarray(X_train), np.asarray(X_test) , len(vocabulory)

def train(X, y, input_shape, output_shape,vocabulory):
	model = Sequential()
	model.add(Embedding(vocabulory, 32, input_length=500))
	model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())	
	model.add(Dense(250, activation='relu'))
	model.add(Dense(22, activation='softmax'))
	model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

	#model.add(Embedding(input_shape+1, 1024))
	#model.add(Dense(1024, activation='relu', input_dim=input_shape))
	#model.add(Dense(512, activation='relu'))
	#model.add(Dense(output_shape, activation='softmax'))
	#model.compile(optimizer='adamax', loss='categorical_hinge', metrics=['accuracy'])
	model.fit(X, y, epochs=1, batch_size=64,validation_split=0.2,verbose=2)
	model.summary()
	model_json = model.to_json()
	with open("./classifiers/nn.json", "w") as json_file:
	    json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("./classifiers/nn.h5")
	print("Saved model to disk")
	return model

def test(model, X, y):
	y_pred = model.predict(X) 
	pred_labels = np.argmax(y_pred, axis=1)
	positives = 0;
	for i in range(y_pred.shape[0]):
		print "prediction: "+str(pred_labels[i])+" truth: "+str(y[i])
		if pred_labels[i] == y[i]:
			positives += 1
	print '--- Accuracy ---: ', float(positives)/float(y_pred.shape[0])
	return pred_labels

if __name__ == '__main__':
	
	labels = 22
	#train_X, train_labels, test_X, test_Y = get_data()
	train_X1=np.load('train_X.npy')
	print (train_X1[:10])
	test_X1=np.load('test_X.npy')
	train_labels=pickle.load(open('train_Y.txt', 'rb'))
	print train_labels[:10]
	test_Y=pickle.load(open('test_Y.txt', 'rb'))
	train_X,test_X , vocabulory=one_hot_vec(train_X1,test_X1)
	train_Y = np_utils.to_categorical(train_labels[:], labels)
	print '--- train X shape: --', train_X.shape
	print '--- train labels ---', train_Y.shape
	print '--- test X shape ---', test_X.shape
	model = train(train_X, train_Y, train_X.shape[1], labels,vocabulory)
	pred = test(model, test_X, test_Y)
