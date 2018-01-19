import numpy as np 
import sys

print('getting file list...')
filelist = list()
for n in range(len(sys.argv)):
	if n == 0:
		continue
	filelist.append(sys.argv[n])


print('drafting answers into array : answer')
answer = list()
for file in filelist:
	answer.append(np.genfromtxt(file, delimiter = ',', dtype=int, skip_header = 1)[:,1])
answer = np.asarray(answer).T
print('get answer list with shape:', answer.shape)



print('start voting')
vote = list()
for n in range(len(answer)):
	vote.append(np.argmax(np.bincount(np.abs(answer[n]))))
vote = np.asarray(vote)
print('save result with shape:', vote.shape)



print('start export answer')

save = open('answer/ensemble.csv', 'w')
save.write('id,ans\n')
for i in range(len(vote)):
	save.write(str(i+1) + ',' + str(vote[i]) + '\n')
save.close() 

