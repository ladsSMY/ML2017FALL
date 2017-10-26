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
x_train = np.concatenate((x_train,x_train**2), axis=1)

y_train = np.genfromtxt(m4, delimiter = ',')
y_train = np.reshape(y_train,(len(y_train),1))
y_train = y_train[1:]
size = len(x_train[0])


'''Define Sigmoid Function'''
def sigmoid(z):
    return np.clip((1/(1+np.exp(-z))), 0.00000000000001, 0.99999999999999)


'''Normalize'''
std = np.std(x_train,axis=0)
mean = np.mean(x_train,axis=0)
x_train = np.divide(np.subtract(x_train,mean),std)


'''Gradient Decent'''
b = 0.1
w = np.full((size,1),0.1)
lr = 0.1
iteration = 20000
pa = iteration/100
b_lr = 0.0
w_lr = np.full((size,1),0.0)


'''Start Training'''
for it in range(iteration):
	est = sigmoid(np.dot(x_train,w)+b)
	err = y_train - est
	b_grad = -2*np.sum(err)
	w_grad = -2*np.dot(np.transpose(x_train),err)

	if it%pa == 0:
		#loss = np.sum(-(y_train*np.log(est)+(1-y_train)*np.log(1-est)))
		print(it/pa,'%')
	
	b_lr = b_lr + b_grad**2
	w_lr = w_lr + w_grad**2

	b = b - lr/np.sqrt(b_lr)*b_grad
	w = w - lr/np.sqrt(w_lr)*w_grad


'''Save Load'''
#np.save('w2.npy',w)
#np.save('b2.npy',b)
#w = np.load('w.npy')
#b = np.load('b.npy')


'''Test'''
x_test = np.genfromtxt(m5, delimiter = ',',skip_header = 1)
x_test = np.concatenate((x_test,x_test**2), axis=1)
x_test = np.divide(np.subtract(x_test,mean),std)
y_test = sigmoid(np.dot(x_test,w)+b) >= 0.5

save = open(m6, 'w')
save.write("id,label\n")
for i in range(len(y_test)):
    save.write(str(i+1) + "," + str(int(y_test[i])) + "\n")
save.close()









