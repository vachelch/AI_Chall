
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

base_dir = 'data/new/'
val_dir = os.path.join(base_dir, 'test_split_search.txt')
label_dir = os.path.join(base_dir, 'val_' + sys.argv[1] +'.txt')
vocab_dir = os.path.join(base_dir, 'cnews.'  + sys.argv[1] + '.txt')


pred_dir = 'data/pred/'
pred_path = os.path.join(pred_dir, 'preds_' + sys.argv[1] + '.npy')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation_' + sys.argv[1])  # 最佳验证结果保存路径


def test():
    print("Loading test data...")
    start_time = time.time()
    x_test, _, _ = process_file(val_dir, label_dir, word_to_id, cat_to_id, config.seq_length)


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


    np.save(pred_path, y_pred_cls)



if __name__ == '__main__':
    config = TCNNConfig()
    
    config.loss_weight = [1 for c in range(4)]
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)

    model = TextCNN(config)
    test()




























        