from collections import Counter
import numpy as np 

def get_loss_weight(y_train_onehot):
	y_train_1d = np.argmax(y_train_onehot, 1)
	ct = Counter(y_train_1d)

	loss_weights = [ct[i] for i in range(len(ct))]
	mx_num = max(loss_weights)

	loss_weights = [mx_num / (cnt + 0.1) for cnt in loss_weights]

	return loss_weights

def transform(x_train, y_train_onehot):
	y_train_1d = np.argmax(y_train_onehot, 1)
	ct = Counter(y_train_1d)
	mx_cls = ct.most_common(1)[0][0]

	index2id = {}
	id2index = {}
	for i in range(4):
		if i != mx_cls:
			index = len(index2id)
			index2id[index] = i
			id2index[i] = index

	y_train_1 = [[1, 0] if label == mx_cls else [0, 1] for label in y_train_1d]

	x_train_2 = []
	for i, row in enumerate(x_train):
		if y_train_1d[i] != mx_cls:
			x_train_2.append(row)

	y_train_2 = []

	for label in y_train_1d:
		if label != mx_cls:
			index = id2index[label]
			onehot = [0 for i in range(3)]
			onehot[index] = 1
			y_train_2.append(onehot)

	return np.array(x_train), np.array(y_train_1), np.array(x_train_2), np.array(y_train_2), index2id


















