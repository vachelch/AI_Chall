from collections import Counter
import os
import pickle

base_dir = 'data/new'
train_split_dir = os.path.join(base_dir, 'train_split_search.txt')
train_label_dir = os.path.join(base_dir, "train_location_traffic_convenience.txt")

mean_rank_dir = os.path.join(base_dir, 'mean_word_rank.pk')

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


def word2rank_func(word2cnt):
	cnt_set = set()
	for word, cnt in word2cnt.items():
		cnt_set.add(cnt)

	cnt_list = sorted(list(cnt_set), reverse = True)

	cnt2rank = dict()
	for i, cnt in enumerate(cnt_list):
		cnt2rank[cnt] = i

	word2rank = Counter()
	for word, cnt in word2cnt.items():
		word2rank[word] = cnt2rank[cnt]

	return word2rank

data = read_data(train_split_dir)
labels = read_label(train_label_dir)

word2cnt = word2cnt_func(data)
word2rank = word2rank_func(word2cnt)

print('mean word2cnt')
print(word2cnt.most_common(30))
print()

print('mean word2rank')
print(word2rank.most_common()[-30:])

pickle.dump(word2rank, open('mean_word_rank.pk', 'wb'))


# specific class
word2cnt_cls = word2cnt_func(data, labels, -1)
word2rank_cls = word2rank_func(word2cnt_cls)


word2rankAlter = Counter()
word2rankAlterAbs = Counter()
for word, rank in word2rank_cls.items():
	if word2rank[word] <= 3000:
		word2rankAlter[word] = (word2rank[word] - rank)*(3001 - rank)
		word2rankAlterAbs[word] = abs(rank - word2rank[word]) * (3001 - rank)


print('word2cnt_cls')
print(word2cnt_cls.most_common(30))
print()

print('word2rank_cls')
print(word2rank_cls.most_common()[-30:])

print()
print("word2rankAlter")
print(word2rankAlter.most_common()[-500:])
print()
print("word2rankAlterAbs")
print(word2rankAlterAbs.most_common()[-500:])












