import sys
s = sys.argv[1]

#read data
data = open( s,'r')

#save to data
for line in data:
	word = line.split()
	
name  = []
count = []
same = 0

for i in range(len(word)):
	same = 0
	if len(name) == 0:
		name.append(word[i])
		continue
	
	for j in range(len(name)):
		if word[i] == name[j]:
			same = 1
			break
			
	if same == 0:
		name.append(word[i])


for x in name:
	i = word.count(x)
	count.append(i)


file = open("Q1.txt", 'w')

for i in range(len(name)):
	file.write(str(name[i]))
	file.write(" ")
	file.write(str(i))
	file.write(" ")
	file.write(str(count[i]))
	if i < (len(name)-1):
		file.write("\n")

			

	