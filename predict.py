#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics

from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab
from helper.utils import transform, get_loss_weight

os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

base_dir = 'data/new/'
pred_dir = 'data/pred/'

vocab_dir = os.path.join(base_dir, 'cnews.vocab_' + sys.argv[2] +'.txt')
save_dir = 'checkpoints/textcnn'
save_path_1 = os.path.join(save_dir, 'best_validation_step1_' + sys.argv[2])  # 最佳验证结果保存路径
save_path_2 = os.path.join(save_dir, 'best_validation_step2_' + sys.argv[2]) 

# train_dir = os.path.join(base_dir, 'train_data.txt')
# test_dir = os.path.join(base_dir, 'val_data.txt')
# val_dir = os.path.join(base_dir, 'val_data.txt')
train_dir = os.path.join(base_dir, 'train_split_search.txt')
train_label_dir = os.path.join(base_dir, 'train_' + sys.argv[2] +'.txt')
test_label_dir = os.path.join(base_dir, 'val_' + sys.argv[2] + '.txt')

# predict val
if sys.argv[1] == 'val':
    test_dir = os.path.join(base_dir, 'val_split_search.txt')
    pred_path = os.path.join(pred_dir, 'preds_test_' + sys.argv[2] + '.npy')

# predict test
elif sys.argv[1] == 'test':
    test_dir = os.path.join(base_dir, 'test_split_search.txt')
    pred_path = os.path.join(pred_dir, 'preds_test_' + sys.argv[2] + '.npy')



def test(x_test, y_test, save_path_1, save_path_2, most_cls, index2id, model1, model2):
    print("Loading test data...")
    start_time = time.time()

    # model1
    session1 = tf.Session()
    session1.run(tf.global_variables_initializer())
    saver1 = tf.train.Saver()
    saver1.restore(sess=session1, save_path=save_path_1)  # 读取保存的模型

    # model2
    session2 = tf.Session()
    session2.run(tf.global_variables_initializer())
    saver2 = tf.train.Saver()
    saver2.restore(sess=session2, save_path=save_path_2)  # 读取保存的模型

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1


    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls_step1 = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    y_pred_cls_step2 = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict1 = {
            model1.input_x: x_test[start_id:end_id],
            model1.keep_prob: 1.0
        }
        feed_dict2 = {
            model2.input_x: x_test[start_id:end_id],
            model2.keep_prob: 1.0
        }

        y_pred_cls_step1[start_id:end_id] = session1.run(model1.y_pred_cls, feed_dict=feed_dict1)
        y_pred_cls_step2[start_id:end_id] = session2.run(model2.y_pred_cls, feed_dict=feed_dict2)

    y_pred_cls = [most_cls if y_pred_cls_step1[i] == 0 else index2id[y_pred_cls_step2[i]] for i in range(len(y_pred_cls_step1))]
    
    if sys.argv[1] == 'val':
    # 评估
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
    if len(sys.argv) != 3 or sys.argv[1] not in ['val', 'test']:
        raise ValueError("""usage: python run_cnn.py [val / test] [label]""")

    print('Configuring CNN model...')

    config1 = TCNNConfig()
    config2 = TCNNConfig()

    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, train_label_dir, vocab_dir, config1.vocab_size)
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)

    # raw data    
    x_train, y_train, train_label_weight = process_file(train_dir, train_label_dir, word_to_id, cat_to_id, config1.seq_length)
    x_test, y_test, test_label_weight = process_file(test_dir, test_label_dir, word_to_id, cat_to_id, config1.seq_length)

    # step 1, classify largest class with other; step 2, classify smaller class
    # x_train_1, y_train_1 are input data of step1, x_train_2 as the same to step2
    most_cls = np.argmin(train_label_weight)
    x_train_1, y_train_1, x_train_2, y_train_2, index2id = transform(x_train, y_train)
    x_test_1, y_test_1, x_test_2, y_test_2, _ = transform(x_test, y_test)

    # model config
    config1.name = "step1cls"
    config2.name = "step2cls"

    config1.num_classes = 2
    config2.num_classes = 3

    config1.loss_weight = get_loss_weight(y_train_1)
    config2.loss_weight = get_loss_weight(y_train_2)

    print(config1.loss_weight)
    print(config2.loss_weight)

    model1 = TextCNN(config1)
    model2 = TextCNN(config2)

    test(x_test, y_test, save_path_1, save_path_2, most_cls, index2id, model1, model2)
































