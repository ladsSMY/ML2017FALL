import sys
import numpy as np
m1 = sys.argv[1]
m2 = sys.argv[2]
m3 = sys.argv[3]
m4 = sys.argv[4]
m5 = sys.argv[5]
m6 = sys.argv[6]

'''Read Train Data'''
x_train = np.genfromtxt(m3, delimiter = ',',skip_header = 1)
y_train = np.genfromtxt(m4, delimiter = ',',skip_header = 1)
y_train = np.reshape(y_train,(len(y_train),1))
f_size = len(x_train[0]) # feature size
d_size = len(x_train) # data size

'''Define Sigmoid Function'''
def sigmoid(z):
    return np.clip((1/(1+np.exp(-z))), 0.00000000000001, 0.99999999999999)


'''Find Mu'''
class1 = 0
class2 = 0
mu1 = np.zeros((1,f_size))
mu2 = np.zeros((1,f_size))
for n in range(d_size):
	if y_train[n] == 1:
		mu1 += x_train[n]
		class1 += 1
	else:
		mu2 += x_train[n]
		class2 += 1
mu1 /= class1
mu2 /= class2


'''Find Sigma'''
sig1 = np.zeros((f_size,f_size))
sig2 = np.zeros((f_size,f_size))
for n in range(d_size):
	if y_train[n] == 1:
		sig1 += np.dot(np.transpose(x_train[n]-mu1),(x_train[n]-mu1))
	else:
		sig2 += np.dot(np.transpose(x_train[n]-mu2),(x_train[n]-mu2))
sig1 /= class1
sig2 /= class2
sig = float(class1)/d_size*sig1 + float(class2)/d_size*sig2 # share sigma
invsig = np.linalg.inv(sig)


'''Get w and b'''
w = np.dot(mu1-mu2,invsig)
b = np.log(float(class1)/class2)-0.5*np.dot(np.dot(mu1,invsig),np.transpose(mu1))+0.5*np.dot(np.dot(mu2,invsig),np.transpose(mu2))
	

'''Test'''
x_test = np.genfromtxt(m5, delimiter = ',',skip_header = 1)
predict = np.around(sigmoid(np.dot(x_test,np.transpose(w))+b))
save = open(m6, 'w')
save.write("id,label\n")
for i in range(len(predict)):
    save.write(str(i+1) + "," + str(int(predict[i])) + "\n")
save.close()








