from gensim.models import word2vec,KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import pandas as pd
import numpy as np
import csv, sys, os
import gensim


jieba.set_dictionary('data/dict.txt.big')
dataFile = open("data/testing_data.csv","r",encoding='UTF-8')
testing_df = pd.read_csv(dataFile)
testing_cut_set = []


def processing_testing_data(data):
	id_ = data['id']
	sentences = []
	sentences2 = []
	q_cut_set = []
	op_cut_set = []
	
	for line in data['dialogue']:
		sentences.append(line.split('\t'))

	for e in sentences:
		reg = []
		for ee in e:
			lines1 = ee.split(':')[1]
			lines = jieba.cut(lines1)
			line_1 = (" ".join(lines))
			reg.append(line_1)
		q_cut_set.append(reg)

	for line2 in data['options']:
		sentences2.append(line2.split('\t'))

	for six in sentences2:
		reg = []
		for one in six:
			word_list = one.split(':')[1]
			cut = jieba.cut(word_list)
			cut_sp = (" ".join(cut))
			ls = cut_sp.split(' ')
			reg.append(ls)
		op_cut_set.append(reg)
	
	return(id_,q_cut_set,op_cut_set)


def get_ans(q,opt,model,cnt):
	
	emb_cnt = 0
	dim = int(sys.argv[1])
	avg_dlg_emb = np.zeros((dim,))
	
	for words in q:
		q_reg = words.split(' ')	
		for word in words.split(' '):
			if word in model.vocab and word != '\t':
				avg_dlg_emb += model[word]
				emb_cnt += 1
	
	if emb_cnt == 0:
		emb_cnt += 1

	avg_dlg_emb /= emb_cnt
	emb_cnt = 0 
	max_sim = -10
	max_idx = -1
	cnt1 = 0

	for choice in opt:
		avg_ans_emb = np.zeros((dim,))
		for word in choice:
			if word in model.vocab:
				avg_ans_emb += model[word]
				emb_cnt += 1
		
		if emb_cnt == 0:
			emb_cnt += 1

		avg_ans_emb /= emb_cnt
		
		# cosine similarity
		sim = 0
		if np.linalg.norm(avg_dlg_emb) !=0 and np.linalg.norm(avg_ans_emb) !=0:
			sim = np.dot(avg_dlg_emb, avg_ans_emb) / np.linalg.norm(avg_dlg_emb) / np.linalg.norm(avg_ans_emb)
		
		if sim>max_sim:
			max_sim = sim
			max_idx = cnt1
		cnt1 +=1

	return max_idx



id_,question_set,options_set = processing_testing_data(testing_df)


w2vModel_path = sys.argv[2] ##
model = KeyedVectors.load(w2vModel_path)


answer = []
for cnt in range(len(question_set)):
	answer.append(get_ans(question_set[cnt],options_set[cnt],model,cnt))

name = sys.argv[3]

print('prediction is done')
f = open('answer/'+name+'.csv','w')
fo = csv.writer(f)
cnt = 1
header = ['id','ans']
fo.writerow(header)
for e in answer:
	s = [cnt,str(e)]
	fo.writerow(s)
	cnt = cnt + 1


