import sys
import numpy as np
import pandas as pd
import keras.backend as K
from keras.models import Model, Sequential, load_model
from keras.layers import add, Dot, Input, Dense, Lambda, Reshape, Dropout, Embedding
from keras.regularizers import l2
from keras.initializers import Zeros
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.engine.topology import Layer
from keras.preprocessing.sequence import pad_sequences


'''

		Global Paths

'''
PATH_test		= sys.argv[1]
PATH_prediction = sys.argv[2]
PATH_movies		= sys.argv[3]
PATH_users		= sys.argv[4]
PATH_train		= 'train.csv'
PATH_model		= 'mf_best.h5'


'''

		Function Defined

'''
def read_data(istrain):
	# get raw data
    train_raw, test_raw = pd.read_csv(PATH_train), pd.read_csv(PATH_test)
    train_raw['test'], test_raw['test'] = 0, 1
    raw = pd.concat([train_raw, test_raw])

    # encode: given new id 
    uid, mid = raw['UserID'].unique(), raw['MovieID'].unique()
    user_id  = {n: id for id, n in enumerate(uid)}
    movie_id = {n: id for id, n in enumerate(mid)}
    raw['UserID']  = raw['UserID'].apply(lambda x: user_id[x])
    raw['MovieID'] = raw['MovieID'].apply(lambda x: movie_id[x])
    np.save('user_id',user_id)
    np.save('movie_id',movie_id)
    # return
    if istrain:
    	train = raw.loc[raw['test'] == 0]
    	return train[['UserID', 'MovieID']].values, train['Rating'].values, len(user_id), len(movie_id)
    
    else:
    	test = raw.loc[raw['test'] == 1]
    	return test[['UserID', 'MovieID']].values

def read_test():
	test_raw	= pd.read_csv(PATH_test)
	user_id		= np.load('user_id.npy')[()]
	movie_id	= np.load('movie_id.npy')[()]
	test_raw['UserID']  = test_raw['UserID'].apply(lambda x: user_id[x])
	test_raw['MovieID'] = test_raw['MovieID'].apply(lambda x: movie_id[x])
	return test_raw[['UserID', 'MovieID']].values

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.pow(y_true - y_pred, 2)))

def normalize(y):
	y_mean = np.mean(y, axis=0)
	y_std = np.std(y, axis=0)	
	y = (y - y_mean)/y_std
	return y_mean, y_std, y

'''

		Main

'''
if __name__ == '__main__':
	'''
	# get train data and randomize
	X_train, Y_train, nuser, nmovie = read_data(istrain=1)

	# normalization
	y_mean, y_std, Y_train = normalize(Y_train)
	
	np.random.seed(943153)
	rand_id = np.random.permutation(len(X_train))
	X_train, Y_train = X_train[rand_id], Y_train[rand_id]

	# training parameters
	dim			= 8 
	epochs		= 1000
	batch_size	= 128
	patience	= 20

	# build model
	# User
	Ui = Input(shape=(1,))
	U = Embedding(nuser, dim, embeddings_regularizer=l2(0.000001))(Ui)
	U = Reshape((dim,))(U)
	U = Dropout(0.15)(U)

	# Movie
	Mi = Input(shape=(1,))
	M = Embedding(nmovie, dim, embeddings_regularizer=l2(0.000001))(Mi)
	M = Reshape((dim,))(M)
	M = Dropout(0.15)(M)

	# bias
	Ub = Reshape((1,))(Embedding(nuser, 1, embeddings_regularizer=l2(0.00001))(Ui))
	Mb = Reshape((1,))(Embedding(nmovie, 1, embeddings_regularizer=l2(0.00001))(Mi))

	# dot-add-out
	DOT = Dot(axes=-1)([U, M])
	ADD = add([DOT, Ub, Mb])
	model = Model(inputs=[Ui, Mi], outputs=[ADD])

	model.summary()
	print('>>>>>>>>>>>>>>>>>>>>>>>>>')
	print('  IN---U')
	print('       |')
	print('      DOT---ADD---OUT')
	print('       |     |')
	print(' OUT---M    bias')
	print('<<<<<<<<<<<<<<<<<<<<<<<<<')
	
	# set callback
	callbacks = [	EarlyStopping(monitor='val_rmse', patience=patience),
  					ModelCheckpoint(NAME+'_long_best.h5', monitor='val_rmse', save_best_only=True)	]

	# compile & fit
	model.compile(loss='mse', optimizer='adam', metrics=[rmse])
	model.fit([X_train[:, 0], X_train[:, 1]], Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=callbacks)
	del model
	'''
	# test
	y_mean = 3.58171208604
	y_std  = 1.11689766115
	X_test = read_test()
	model = load_model(PATH_model, custom_objects={'rmse': rmse})
	predict = model.predict([X_test[:, 0], X_test[:, 1]]).squeeze()

	predict = (predict * y_std + y_mean).clip(1.0, 5.0)

	save = open(PATH_prediction, 'w')
	save.write("TestDataID,Rating\n")
	for i in range(len(predict)):
		save.write(str(i+1)+","+str(float(predict[i]))+"\n")
	save.close()