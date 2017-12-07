from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, Conv1D, MaxPooling1D
from keras.utils  import np_utils, to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import sequence

import _pickle as pk
import numpy as np 
import pandas as pd
import sys

def LoadData(istrainset):
	if istrainset:
		rawdata = pd.read_csv(PATH_training_label, sep = '\+\+\+\$\+\+\+')
		rawdata = rawdata.as_matrix()
		rawdata = rawdata.T
		X = rawdata[1]
		Y = np_utils.to_categorical(rawdata[0], 2)
		return X, Y
	else:
		file = open(PATH_testing_data, 'r')
		rawdata = list()
		for line in open(PATH_testing_data):
			line = file.readline()
			rawdata.append(line)
		file.close()
		del rawdata[0]
		dot = int()
		for n in range(len(rawdata)):
			rawdata[n] = rawdata[n].split('\n')[0]
			dot = rawdata[n].index(',')+1
			rawdata[n] = rawdata[n][dot:]
		return rawdata



'''Global Path'''
PATH_testing_data   = sys.argv[1] 
PATH_prediction     = sys.argv[2]



'''define parameters'''
# Token
num_words = 40000
filters = ''

# Embedding
max_features = 20000
maxlen = 100
embedding_size = 128

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 30
epochs = 2



'''Prediction 1'''
x_test = LoadData(0)
model1 = load_model('model_1.h5')
token = pk.load(open('model_1_tokenize', 'rb'))
x_test = token.texts_to_sequences(x_test)
x_test = pad_sequences(x_test, maxlen=maxlen) 
predict1 = model1.predict(x_test)
print('predict 1 complete')



'''Prediction 2'''
def GetTestSet(dictionary):
	file = open(PATH_testing_data, 'r')
	rawdata = list()
	for line in open(PATH_testing_data):
		line = file.readline()
		rawdata.append(line)
	file.close()

	# kill header
	del rawdata[0]

	# kill /n and id
	dot = int()
	for n in range(len(rawdata)):
		rawdata[n] = rawdata[n].split('\n')[0]
		dot = rawdata[n].index(',')+1
		rawdata[n] = rawdata[n][dot:]
		rawdata[n] = rawdata[n].split()
		for m in range(len(rawdata[n])):
			if rawdata[n][m] in dictionary.keys():
				rawdata[n][m] = dictionary[rawdata[n][m]]
			else:
				rawdata[n][m] = dictionary['NotInDict']
	return rawdata

model2 = load_model('model_2.h5')
dictionary = np.load('model_2_tokenize')

x_test = GetTestSet(dictionary)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

predict2 = model2.predict(x_test)
print('predict 2 complete')



'''Prediction 3'''
x_test = LoadData(0)
model3 = load_model('model_3.h5')
token = pk.load(open('model_3_tokenize', 'rb'))

x_test = token.texts_to_sequences(x_test)
x_test = pad_sequences(x_test, maxlen=maxlen) 
predict3 = model3.predict(x_test)
print('predict 3 complete')



'''Merge'''
save = open(PATH_prediction, 'w')
predict = predict1 + predict2 +predict3
save.write("id,label\n")
for i in range(len(predict)):
	save.write(str(i)+","+str(int(np.argmax(predict[i])))+"\n")
save.close()
