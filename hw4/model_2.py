from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.utils  import np_utils
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
def ParseTrainingLabel():
	rawdata = pd.read_csv(PATH_training_label, sep = '\+\+\+\$\+\+\+')
	rawdata = rawdata.as_matrix()
	rawdata = rawdata.T
	label = rawdata[0]
	sentence = rawdata[1]

	for n in range(len(sentence)):
		sentence[n] = sentence[n].split()

	label = np_utils.to_categorical(label, 2)

	return label, sentence

def BuildDictionary(sentence):
	dictionary = { }
	wordcount = 0
	for line in sentence:
		for n in range(len(line)):
			if line[n] in dictionary.keys():
				line[n] = dictionary[line[n]]
			else:
				newpage = {line[n]:wordcount}
				wordcount = wordcount+1
				dictionary.update(newpage)
				line[n] = dictionary[line[n]]
	otherpage = {'NotInDict':wordcount}
	dictionary.update(otherpage)
	print('sentences count:', len(sentence))
	print('words count:', wordcount)
	return sentence, dictionary, wordcount



'''Training Stuff'''
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



'''MAIN'''
label, sentence = ParseTrainingLabel()
sentence, dictionary, nid = BuildDictionary(sentence)
pk.dump(dictionary, open(PATH_token, 'wb'))


# spliting valid set
x_train, y_train = sentence[30000:], label[30000:]
x_valid, y_valid = sentence[:30000], label[:30000]

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_valid = sequence.pad_sequences(x_valid, maxlen=maxlen)

model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Dropout(0.25))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs)

model.save(PATH_model)

score, acc = model.evaluate(x_valid, y_valid, batch_size=batch_size)
print('Valid score:', score)
print('Valid accuracy:', acc)








