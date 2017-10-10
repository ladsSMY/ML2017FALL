import numpy as np
from numpy import genfromtxt
from numpy import *
import sys
input_ = sys.argv[1]
output_ = sys.argv[2]


'''Load in testdata'''
testdata = genfromtxt(input_, delimiter=',')
model = genfromtxt('hw1.model', delimiter=',')
(height,width) = testdata.shape



'''Filt data from testdata'''
data = np.zeros((18,240*9))
curd = 0 # current data
curt = 0 # current time
for h in range(0,height):
	curt_save = curt
	for w in range(2,width):
		data[curd][curt] = testdata[h][w]
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


'''save to output_'''
save = open(output_, 'w')
save.write('id,value\n')

t_ = 9 #time
d_ = 3 #data
p_ = 2 #pm2.5

b = model[9][0]
w = model[:9]

time = 0
while time < 240*9:
	p = b+(w*partdata[time:time+t_]).sum()
	id_ = time/9
	save.write('id_')
	save.write(str(int(id_)))
	save.write(',')
	save.write(str(p))
	save.write('\n')
	time += 9
