import csv 
import numpy as np
from numpy.linalg import inv
from numpy import genfromtxt
from numpy import *
import random
import math
import sys
input_ = sys.argv[1]
output_ = sys.argv[2]

'''READ DATA'''
data = []
for i in range(18):
	data.append([])

n_row = 0
text = open('train.csv', 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
    if n_row != 0:
        for i in range(3,27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0))	
    n_row = n_row+1
text.close()


'''PARSE DATA TO (X,Y)'''
x = []
y = []
# 12 months are not continues
for i in range(12):
    for j in range(471):
        x.append([])
        # 18 types 
        for t in range(7,10): #onlyuu want type 7 to 9
            # 9 hours
            for s in range(9):
                x[471*i+j].append(data[t][480*i+j+s] )
        y.append(data[9][480*i+j+9])
x = np.array(x)
y = np.array(y)

# add square term
x = np.concatenate((x,x**2), axis=1)

# add bias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)


'''INIT HYPERPARAMS'''
w = np.zeros(len(x[0]))
l_rate = 10
repeat = 20000


'''CLOSE FORM SOLUTION'''
w = np.matmul(np.matmul(inv(np.matmul(x.transpose(),x)),x.transpose()),y)


'''TRAINING
x_t = x.transpose()
s_gra = np.zeros(len(x[0]))

for i in range(repeat):
    hypo = np.dot(x,w)
    loss = hypo - y
    cost = np.sum(loss**2) / len(x)
    cost_a  = math.sqrt(cost)
    gra = np.dot(x_t,loss)
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    w = w - l_rate * gra/ada
    print ('iteration: %d | Cost: %f  ' % ( i,cost_a))'''


'''SAVE/READ MODEL'''
np.save('model.npy',w)
w = np.load('model.npy')


'''TEST'''
test_x = []
n_row = 0
text = open(input_ ,"r")
row = csv.reader(text , delimiter= ",")

for r in row:
    if n_row %18 == 0:
        test_x.append([])
        for i in range(2,11):
            test_x[n_row//18].append(float(r[i]) )
    else :
        for i in range(2,11):
            if r[i] !="NR":
                test_x[n_row//18].append(float(r[i]))
            else:
                test_x[n_row//18].append(0)
    n_row = n_row+1
text.close()
test_x = np.array(test_x)

# add square term
test_x = np.concatenate((test_x,test_x**2), axis=1)

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)

test_x = np.transpose(test_x)
temp_x = np.concatenate((np.transpose(test_x[0:1]), 
                         np.transpose(test_x[64:91]),
                         np.transpose(test_x[226:253])), axis = 1)####
test_x = temp_x


'''ANSWER'''
ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    ans[i].append(a)

text = open(output_, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()
