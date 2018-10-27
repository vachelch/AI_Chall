from collections import Counter
import os, math
import numpy as np 

# base_dir = 'data/new'
# train_split_dir = os.path.join(base_dir, 'train_split_search.txt')


def read_data(train_split_dir):
	data = []
	with open(train_split_dir, 'r', encoding='utf8') as f:
		for i, doc in enumerate(f.readlines()):
			data.append(doc.split())

	return data

def read_label(train_label_dir):
	labels = []
	with open(train_label_dir, 'r') as f:
		for i, doc in enumerate(f.readlines()):
			labels.append(int(doc))


	return labels

def word2cnt_func(data, labels = None, label = None):
	word2cnt = Counter()

	if labels:
		for i, row in enumerate(data):
			if labels[i] == label:
				for word in row:
					word2cnt[word] += 1
	else:
		for i, row in enumerate(data):
			for word in row:
				word2cnt[word] += 1;

	return word2cnt


def get_tf_idf_word(word2cnt, word2cnt_cls):
	tf_idf = Counter()
	for word, cnt in word2cnt_cls.items():
		tf = cnt
		idf = math.log(word2cnt[word]/(word2cnt[word] - cnt + 0.1))

		tf_idf[word] = tf*idf

	return tf_idf

def get_tf_idf_doc(word2cnt, word2cnt_cls, data, labels):
	tf_idf = Counter()
	word2docNum = Counter()
	total_doc_num = 1

	for i, row in enumerate(data):
		if labels[i] != -2:
			total_doc_num += 1
			words_st = set(row)
			for word in words_st:
				word2docNum[word] += 1

	for word, cnt in word2cnt_cls.items():
		tf = cnt
		idf = math.log(total_doc_num/(word2docNum[word] + 1))

		tf_idf[word] = tf*idf

	return tf_idf

def top_words(train_split_dir, train_label_dir, vocab_size):
	data = read_data(train_split_dir)
	labels = read_label(train_label_dir)

	top_labels = Counter(labels)
	top_labels, _ = zip(*(top_labels.most_common()[-3:]))

	word2cnt = word2cnt_func(data)

	word2cnt_clses = []
	clses = top_labels
	print("choose dictionary from 3 class which have least number data:": clses)

	for label in clses:
		word2cnt_clses.append(word2cnt_func(data, labels, label))

	seed_words_clses = []
	for word2cnt_cls in word2cnt_clses:
		tf_idf = get_tf_idf_word(word2cnt, word2cnt_cls)
		seed_words_cls, _ = list(zip(*(tf_idf.most_common())))
		seed_words_clses.append(seed_words_cls)

	words = set()
	for word_arr in zip(seed_words_clses[0], seed_words_clses[1], seed_words_clses[2]):
		for word in word_arr:
			words.add(word)
			if len(words) == vocab_size:
				return list(words)



























