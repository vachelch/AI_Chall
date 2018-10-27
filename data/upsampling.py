from collections import Counter

def up_sample(contents, labels):
	labels_cnt = Counter(labels)

	least_cls, _ = list(zip(*(labels_cnt.most_common()[-2:])))
	print(least_cls)

	contents0 = []
	contents1 = []
	for content, label in zip(contents, labels):
		if least_cls[0] == label:
			contents0.append(content)
		if least_cls[1] == label:
			contents1.append(content)

	# copy 10 time data of less data
	tmp0 = contents0[:]
	for i in range(9):
		contents0.extend(tmp0)
	labels0 = [least_cls[0] for i in range(len(contents0))]

	tmp1 = contents1[:]
	for i in range(9):
		contents1.extend(tmp1)
	labels1 = [least_cls[1] for i in range(len(contents1))]

	# append to raw data
	contents_up = contents
	labels_up = labels

	contents_up.extend(contents0)
	contents_up.extend(contents1)
	labels_up.extend(labels0)
	labels_up.extend(labels1)

	return contents_up, labels_up


