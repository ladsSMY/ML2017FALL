from gensim.models import word2vec
import jieba
from timeit import default_timer as time
from time import ctime as clk
import sys
from os import listdir
from os.path import isfile, join


print('>>>>>>>>>> start <<<<<<<<<<')
print('start time:', clk())
t1 = time()

trainpath = sys.argv[1]
dataFile = list()
files = listdir(trainpath)
for f in files:
	fullpath = join(trainpath, f)
	if isfile(fullpath):
		if f.split('.')[1] == 'txt':
			dataFile.append(fullpath)


jieba.set_dictionary('data/dict.txt.big')
nonStopContent = []
stopdata = 'data/stopwords.txt'
stopWords = []
with open(stopdata,'r') as stop:
	for line in stop:
		line=line.strip('\n')
		stopWords.append(line)


print('process forward to: filter nonStopWords')
article_num = 0
for i in range(len(dataFile)):
	with open(dataFile[i],'r') as content: 
		print('open data file:',dataFile[i])
		for line in content:
			lines = jieba.cut(line)
			lines_1 = (" ".join(lines))
			if lines_1 not in stopWords:
				nonStopContent.append(lines_1)
			article_num += 1


for e in stopWords:
	jieba.del_word(e)
print('process forward to: save nonStopWords.txt')


nonStopContent_path = 'data/w2v_source.txt'
with open(nonStopContent_path,'w') as f:
	for row in nonStopContent:
		f.write(str(row))			


#word2vec
nonStopContent_path = 'data/w2v_source.txt'
print('process forward to: word2vec')
sentences = word2vec.Text8Corpus(nonStopContent_path)


##########################################################

dim				= int(sys.argv[2]) 
windows			= int(sys.argv[3]) 
min_count		= 1
iteration		= int(sys.argv[4]) 

name = 'dim'+str(dim)+'win'+str(windows)+'min'+str(min_count)+'iter'+str(iteration)

##########################################################

w2vModel= word2vec.Word2Vec(sentences,
							size=dim, 
							window=windows, 
							min_count=min_count,
							iter=iteration,
							sg = 1)
word_vectors = w2vModel.wv

print(w2vModel)
# Save our model.
print('process forward to: save word2vec model')
w2vModel_path = 'w2vmodel/'+name+'.txt'
word_vectors.save(w2vModel_path)

t2 = time()
print('cost %is' %(t2-t1), 'with %i iteration\n\n'%iteration)
