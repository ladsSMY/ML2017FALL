import matplotlib
import matplotlib.pyplot as plt
import random as random
import timeit
import numpy as np
from numpy import genfromtxt
from numpy import *
import sys
input_ = sys.argv[1]


'''Load in traindata'''
traindata = genfromtxt(input_, delimiter=',')
(height,width) = traindata.shape # 4321,27


'''Filt data from traindata'''
data = np.zeros((18,240*24))
curd = 0 # current data
curt = 0 # current time
for h in range(1,height):
	curt_save = curt
	for w in range(3,width):
		data[curd][curt] = traindata[h][w]
		curt += 1
	curd += 1
	if curd == 18:
		curd = 0
	else:
		curt = curt_save

where_are_NaNs = isnan(data)
data[where_are_NaNs] = 0


'''Get part data (ver.0)'''
partdata = data[7:10]
partdata = np.transpose(partdata)


'''Drawing Elements'''
ITER = []
LOSS = []


'''Set training parameters'''
t_ = 9 #time
d_ = 3 #data
p_ = 2 #pm2.5

b = -100
w = -1 * np.ones((t_,d_)) # time=9 data=1
lr_b = 0
lr_w = np.zeros((t_,d_)) #learning rate for each parameters
lr = 1
iteration = 50000
pa = iteration/100


'''Start training'''
start = timeit.default_timer()
print('Start training, iterations =',iteration)
for it in range(iteration):
	grad_b = 0
	grad_w = np.zeros((t_,d_))
	
	# update grad
	for time in range(0,230*24): 
		if time+t_ >= 230*24:
			break

		# temp = ans - bias- all(weight x data) 
		temp = partdata[time+t_][p_] - b - (w*partdata[time:time+t_]).sum()
		grad_b = grad_b - 2.0*temp*1.0
		grad_w = grad_w - 2.0*temp*partdata[time:time+t_]

	#b = b - lr*grad_b
	#w = w - lr*grad_w
	
	# Update learning rate
	lr_b = lr_b + grad_b**2
	lr_w = lr_w + grad_w**2

	# Update parameters
	b = b - lr/np.sqrt(lr_b)*grad_b
	w = w - lr/np.sqrt(lr_w)*grad_w

	# Loss
	loss = 0
	error = 0
	for time in range(0,230*24):
		if time+9 >= 230*24:
			break
		error = partdata[time+t_][p_] - b - (w*partdata[time:time+t_]).sum()
		loss = loss + error**2
	loss = np.sqrt(loss/(230*24))

	if it % 100 == 0:
		if it>10:
			ITER.append(it)
			LOSS.append(loss)

	if it % pa == 0:
		print(it/pa,'%',loss)

stop = timeit.default_timer()
print('train time:',stop - start )
plt.plot(ITER,LOSS)
plt.show()


'''Save model'''
print('Save model to hw1.model')
save = open('hw1.model', 'w')
for i in range(0,t_):
	for j in range(0,d_):
		save.write(str(w[i][j]))
		if j<d_-1:
			save.write(',')
	save.write('\n')

for n in range(0,d_):
	save.write(str(b))
	if n<d_-1:
		save.write(',')

# Valid
print('Start trying valid.....')
ANS = [ ]
PDT = [ ]

for time in range(230*24,240*24): # input datas, cut last 10 days for valid
	if time+t_ >= 240*24:
		break
	ANS.append(partdata[time+t_][p_])
	PDT.append(b+(w*partdata[time:time+t_]).sum())
plt.scatter(ANS,PDT)	
plt.show()

DIF = np.abs(np.subtract(ANS,PDT))
dif = (DIF.sum())/len(DIF)
print(dif)

