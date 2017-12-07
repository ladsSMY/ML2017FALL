from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, Conv1D, MaxPooling1D
from keras.utils  import np_utils, to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.normalization import BatchNormalization

import _pickle as pk
import numpy as np 
import pandas as pd
import sys



'''Global Path'''
NAME                = sys.argv[0].split('.')[0]
PATH_training_label = sys.argv[1] 
PATH_train_nolabel  = sys.argv[2]
PATH_model          = NAME+'.h5'
PATH_token          = NAME+'_tokenize'



'''Function Define'''
def LoadData(istrainset):
	if istrainset:
		rawdata = pd.read_csv(PATH_training_label, sep = '\+\+\+\$\+\+\+')
		rawdata = rawdata.as_matrix()
		rawdata = rawdata.T
		X = rawdata[1]
		Y = np_utils.to_categorical(rawdata[0], 2)
		return X, Y
	else:
		file = open(PATH_testing, 'r')
		rawdata = list()
		for line in open(PATH_testing):
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



'''MAIN'''

# get data
x_train, y_train = LoadData(1)

# define parameters
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

#token
token = Tokenizer(num_words=40000, filters='')
token.fit_on_texts(x_train)

x_train = token.texts_to_sequences(x_train)
x_train = pad_sequences(x_train, maxlen=maxlen) 

pk.dump(token, open(PATH_token, 'wb'))

#build model
model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Dropout(0.5))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(2))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

#train model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs)
model.save(PATH_model)





