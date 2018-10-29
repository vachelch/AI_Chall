
# coding: utf-8

from __future__ import print_function

import os, sys, time
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.keras as kr

from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_category, read_vocab, read_file, process_file
from sklearn import metrics

try:
    bool(type(unicode))
except NameError:
    unicode = str

os.environ["CUDA_VISIBLE_DEVICES"] = "0"    
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation_' + sys.argv[2])  # 最佳验证结果保存路径

base_dir = 'data/new/'
pred_dir = 'data/pred/'

vocab_dir = os.path.join(base_dir, 'cnews.vocab_' + sys.argv[2] +'.txt')
label_dir = os.path.join(base_dir, 'val_' + sys.argv[2] +'.txt')
# predict val
if sys.argv[1] == 'val':
    test_dir = os.path.join(base_dir, 'val_split_search.txt')
    pred_path = os.path.join(pred_dir, 'preds_val_' + sys.argv[2] + '.npy')
    data_dir = 'data/raw/val/sentiment_analysis_validationset.csv'

# predict test
if sys.argv[1] == 'test':
    test_dir = os.path.join(base_dir, 'test_split_search.txt')
    pred_path = os.path.join(pred_dir, 'preds_test_' + sys.argv[2] + '.npy')



def test():
    print("Loading test data...")
    start_time = time.time()
    x_test, _, _ = process_file(test_dir, label_dir, word_to_id, cat_to_id, config.seq_length)


    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    if sys.argv[1] == 'val':
        #y
        data_df = pd.read_csv(data_dir, header=0, encoding='utf8')
        y_test = list(data_df[sys.argv[2]])
        y_test_cls = [cat_to_id[str(y)] for y in y_test]

        # 评估
        print(sys.argv[2])
        print("Precision, Recall and F1-Score...")
        report = metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories)
        print(report)


        # 混淆矩阵
        print("Confusion Matrix...")
        cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
        print(cm)

    if sys.argv[1] == 'test':
        np.save(pred_path, y_pred_cls)



if __name__ == '__main__':
    config = TCNNConfig()
    
    config.loss_weight = [1 for c in range(4)]
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)

    model = TextCNN(config)
    test()




























        