import sys
import numpy as np
import keras.models
from keras.models import load_model


ad_testset = sys.argv[1]
ad_model = 'cnn.h5'
ad_export = sys.argv[2]

def read_testset():
	#load in by genfromtxt
	dataset = np.genfromtxt(fname=ad_testset,
							dtype='str',
							delimiter=' ',
							skip_header=1)
	dataset = dataset.T
	#np.save('testset.npy',dataset)
	
	#fix data into label & feature
	fixdata = dataset[0]
	label = np.zeros(len(fixdata))
	fix = np.zeros(len(fixdata))
	for n in range(len(fixdata)):
		label[n], fix[n] = fixdata[n].split(',')[0], fixdata[n].split(',')[1]
	dataset[0] = fix
	for i in range(2304):
		for j in range(7178):
			dataset[i][j] = float(dataset[i][j])/255
	feature = np.reshape(dataset.T,(7178,48,48,1))
	
	#save feature and label as npy
	#np.save('testfeature.npy',feature)
	
	return feature

	
def export(label, address):
	save = open(address,'w')
	save.write("id,label\n")
	for i in range(len(label)):
		save.write(str(i)+","+str(int(np.argmax(label[i])))+"\n")
	save.close()



'''Main'''
testfeature = read_testset()
model = load_model(ad_model)
testlabel = model.predict(testfeature)
export(testlabel, ad_export)