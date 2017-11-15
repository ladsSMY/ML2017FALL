import numpy as np
import matplotlib.pyplot as plt
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, AveragePooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization


'''GPU Limition'''
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
def get_session(gpu_fraction=0.8):
		
	num_threads = os.environ.get('OMP_NUM_THREADS')
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
	
	if num_threads:
		return tf.Session(config=tf.ConfigProto(
			gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
	else:
		return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


'''Addresses'''
ad_dataset = sys.argv[1]


'''Function Define'''
def from_dataset():
	#load in by genfromtxt
	dataset = np.genfromtxt(fname=ad_dataset,
							dtype='str',
							delimiter=' ',
							skip_header=1)
	dataset = dataset.T
	#np.save('dataset.npy',dataset)
		
	#fix data into label & feature
	fixdata = dataset[0]
	label = np.zeros(len(fixdata))
	fix = np.zeros(len(fixdata))
	for n in range(len(fixdata)):
		label[n], fix[n] = fixdata[n].split(',')[0], fixdata[n].split(',')[1]
	dataset[0] = fix
	for i in range(2304):
		for j in range(28709):
			dataset[i][j] = float(dataset[i][j])/255
	feature = np.reshape(dataset.T,(28709,48,48,1))
	label = np_utils.to_categorical(label, 7)

	#save feature and label as npy
	#np.save('feature.npy',feature)
	#np.save('label.npy',label)

	return feature, label

def from_npy():
	return np.load('feature.npy'), np.load('label.npy')

def read_dataset(n):
	if n == 1:
		return from_dataset()
	else:
		return from_npy()


'''MAIN'''

#limit gpu
#KTF.set_session(get_session())

#read data
feature, label = read_dataset(1)

#cut valid set
featureV, labelV = feature[:3000], label[:3000]
feature, label = feature[3000:], label[3000:]

#build model
if 1+1 == 2:
	model = Sequential()

	model.add(Conv2D(32, (3,3), input_shape=(48,48,1), activation='relu'))
	model.add(Conv2D(32, (3,3), activation='relu'))
	model.add(Conv2D(64, (3,3), activation='relu'))
	model.add(Conv2D(64, (3,3), activation='relu'))
	model.add(MaxPooling2D((2,2)))
	model.add(Dropout(0.5))

	model.add(Conv2D(128, (3,3), activation='relu'))
	model.add(Conv2D(128, (3,3), activation='relu'))
	model.add(MaxPooling2D((2,2)))
	model.add(Dropout(0.5))

	model.add(Conv2D(256, (3,3), activation='relu'))
	model.add(Conv2D(256, (3,3), activation='relu'))
	model.add(MaxPooling2D((2,2)))
	model.add(Dropout(0.5))

	model.add(Flatten())

	model.add(Dense(1024))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(512))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(1024))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(512))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(units=7,activation='softmax'))
	model.summary()

#using adamax
model.compile(loss='categorical_crossentropy',
			  optimizer="adamax",
			  metrics=['accuracy'])

#fit with using image data generator
generate = ImageDataGenerator(rotation_range=15,
							  width_shift_range=0.1,
							  height_shift_range=0.1,
							  horizontal_flip=True)
generate.fit(feature)
record = model.fit_generator(generate.flow(feature, label, batch_size=128),
							 validation_data=(featureV, labelV),
							 steps_per_epoch=len(feature)/5,
							 epochs=20)

#save model and pic
model.save('cnn.h5')