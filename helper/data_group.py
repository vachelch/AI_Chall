import csv
import numpy as np

# 開啟 CSV 檔案
def read_file(raw_path):
	with open(raw_path, 'r', encoding='utf8', newline='') as csvfile:
		rows = csv.reader(csvfile)
		contents = []
		labels = []

		for i, row in enumerate(rows):
			if i == 0:
				label_names = row[2:]
			else:
				content = row[1].replace('\n', '').replace('\t', '').replace('\r', '').replace('\u3000', '')
				contents.append(content)
				labels.append(row[2:])

	return contents, labels, label_names

def write_file(data, file_path):
	with open(file_path, 'w', encoding = 'utf8') as f:
		for row in data:
			f.write(row + '\n')


def save_file(raw_path, new_file):
	contents, labels, label_names = read_file(raw_path)
	labels = np.array(labels)

	write_file(contents, new_file + '_data.txt')
	write_file(label_names, new_file + '_label_names.txt')	
	
	for i, name in enumerate(label_names):
		write_file(labels[:, i], new_file + '_' + name + '.txt')

if __name__ == '__main__':
	train_raw_path = 'data/raw/train/sentiment_analysis_trainingset.csv'
	train_new_file = 'data/new/train'

	val_raw_path = 'data/raw/val/sentiment_analysis_validationset.csv'
	val_new_file = 'data/new/val'


	# test_raw_path = 'data/raw/test/sentiment_analysis_testa.csv'
	# test_new_file = 'data/new/val'

	save_file(train_raw_path, train_new_file)
	save_file(val_raw_path, val_new_file)






























