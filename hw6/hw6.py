import numpy as np
import sys 

pic, test, pred = sys.argv[1], sys.argv[2], sys.argv[3]

label = np.load('DNNX_labels.npy') 
test = np.genfromtxt(test, delimiter = ',', dtype=int, skip_header = 1)
i1, i2 = test[:,1], test[:,2]

save = open(pred, 'w')
save.write("ID,Ans\n")
for i in range(len(test)):
	if label[i1[i]] == label[i2[i]]:
		ans = 1
	else:
		ans = 0
	save.write(str(i) + "," + str(ans) + "\n")
save.close() 
