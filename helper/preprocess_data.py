import jieba
import re, os, time
from data_group import read_file, write_file

train_dir = 'data/raw/train/sentiment_analysis_trainingset.csv'
val_dir = 'data/raw/val/sentiment_analysis_validationset.csv'
test_dir = 'data/raw/test/sentiment_analysis_testa.csv'

def jiaba_split(data):
	data_split = []

	for i, doc in enumerate(data):
		seqs = re.split(r'[;,.，。；\s]\s*', doc)
		row = []
		for seq in seqs:
			text = " ".join(jieba.cut_for_search(seq))
			row.append(text)

		data_split.append(" , ".join(row))

	return data_split

def write_file(data, file_path):
	with open(file_path, 'w', encoding = 'utf8') as f:
		for row in data:
			f.write(row + '\n')

base_dir = 'data/new/'

start = time.time()
train_split_dir = os.path.join(base_dir, 'train_split_search.txt')
test_split_dir = os.path.join(base_dir, 'test_split_search.txt')
val_split_dir = os.path.join(base_dir, 'val_split_search.txt')


train_data, _, _ = read_file(train_dir)
test_data, _, _ = read_file(test_dir)
val_data, _, _ = read_file(val_dir)

train_split = jiaba_split(train_data)
test_split = jiaba_split(test_data)
val_split = jiaba_split(val_data)


write_file(train_split, train_split_dir)
write_file(test_split, test_split_dir)
write_file(val_split, val_split_dir)

print(time.time() - start)
# res = jieba.cut_for_search(test)
# print(" ".join(res))











