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
# train_dir = os.path.join(base_dir, 'train_data.txt')
# test_dir = os.path.join(base_dir, 'val_data.txt')
# val_dir = os.path.join(base_dir, 'val_data.txt')
train_dir = os.path.join(base_dir, 'train_split_search.txt')
test_dir = os.path.join(base_dir, 'val_split_search.txt')
val_dir = os.path.join(base_dir, 'val_split_search.txt')
# vocab_dir = os.path.join(base_dir, 'cnews.vocab_search.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab_' + sys.argv[2] +'.txt')

train_label_dir = os.path.join(base_dir, 'train_' + sys.argv[2] +'.txt')
val_label_dir = os.path.join(base_dir, 'val_' + sys.argv[2] + '.txt')
test_label_dir = val_label_dir

save_dir = 'checkpoints/textcnn'
save_path_1 = os.path.join(save_dir, 'best_validation_step1_' + sys.argv[2])  # 最佳验证结果保存路径
save_path_2 = os.path.join(save_dir, 'best_validation_step2_' + sys.argv[2]) 

pred_dir = 'data/pred/'
pred_path = os.path.join(pred_dir, 'preds_val_' + sys.argv[2] + '.npy')

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob, model):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_, model):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0

    total_labels = []
    total_preds = []

    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0, model)
        loss = sess.run([model.loss], feed_dict=feed_dict)
        # total_loss += loss * batch_len
        # total_acc += acc * batch_len
        loss, val_pred = sess.run([model.loss, model.y_pred_cls], feed_dict=feed_dict)
        total_loss += loss

        for p in y_batch:
        	total_labels.append(p)
        for p in val_pred:
        	total_preds.append(p)

    score = metrics.f1_score(np.argmax(total_labels, 1), total_preds, average = "macro")

    return total_loss / data_len, score


def train(x_train, y_train, x_val, y_val, save_path, model, config):
    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 500  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob, model)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, pred_train = session.run([model.loss, model.y_pred_cls], feed_dict=feed_dict)

                acc_train = metrics.f1_score(np.argmax(y_batch, 1), pred_train, average = 'macro')

                loss_val, acc_val = evaluate(session, x_val, y_val, model)  # todo

                time_dif = get_time_dif(start_time)
                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

                # saver.save(sess=session, save_path=save_path)
                # msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                #       + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5}'
                # print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif))

            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break


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
    np.save(pred_path, y_pred_cls)
    
    # 评估
    print("Precision, Recall and F1-Score...")
    report = metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories)
    print(report)


    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    with open('log.txt', 'a') as f:
        f.write("two step, vocab_size: %d,seed_cls_size: %d\n"%(config1.vocab_size, 3))
        f.write("%s\n"%sys.argv[2])
        f.write("%s\n"%report)
        f.write("%s\n"%cm)
        f.write("---------------------\n")

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python run_cnn.py [train / test] [label]""")

    print('Configuring CNN model...')

    config1 = TCNNConfig()
    config2 = TCNNConfig()

    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, train_label_dir, vocab_dir, config1.vocab_size)
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)

    # raw data    
    x_train, y_train, train_label_weight = process_file(train_dir, train_label_dir, word_to_id, cat_to_id, config1.seq_length)
    x_val, y_val, val_label_weight = process_file(val_dir, val_label_dir, word_to_id, cat_to_id, config1.seq_length)

    print('train_label_weight: ', train_label_weight)
    print('val_label_weight: ', val_label_weight)

    print(np.array(x_train).shape)
    print(np.array(y_train).shape)

    # step 1, classify largest class with other; step 2, classify smaller class
    # x_train_1, y_train_1 are input data of step1, x_train_2 as the same to step2
    most_cls = np.argmin(train_label_weight)
    x_train_1, y_train_1, x_train_2, y_train_2, index2id = transform(x_train, y_train)
    x_val_1, y_val_1, x_val_2, y_val_2, _ = transform(x_val, y_val)

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

    if sys.argv[1] == 'train':
        train(x_train_1, y_train_1, x_val_1, y_val_1, save_path_1, model1, config1)
        train(x_train_2, y_train_2, x_val_2, y_val_2, save_path_2, model2, config2)
    else:
        test(x_val, y_val, save_path_1, save_path_2, most_cls, index2id, model1, model2)
































