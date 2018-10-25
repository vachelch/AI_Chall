
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


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class_name_list = ['1', '0', '-1', '-2']
num_classes = 4


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

base_dir = 'data/new/'
val_dir = os.path.join(base_dir, 'val_split_search.txt')
label_dir = os.path.join(base_dir, 'val_' + sys.argv[1] +'.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab_' + sys.argv[1] +'.txt')


pred_dir = 'data/pred/'
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)
pred_path = os.path.join(pred_dir, 'preds_val_' + sys.argv[1] + '.npy')

result_analyse_dir = './val_analyse'
if not os.path.exists(result_analyse_dir):
    os.mkdir(result_analyse_dir)
result_fp_list = []
for i in range(num_classes):
    result_fp_list.append(os.path.join(result_analyse_dir, sys.argv[1] + '_' + class_name_list[i] + '.csv'))



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

    #y
    data_df = pd.read_csv('data/raw/val/sentiment_analysis_validationset.csv', header=0, encoding='utf8')
    y_test = list(data_df[sys.argv[1]])
    y_test_cls = [cat_to_id[str(y)] for y in y_test]

    # 评估
    print(sys.argv[1])
    print("Precision, Recall and F1-Score...")
    report = metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories)
    print(report)


    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    with open('log.txt', 'a') as f:
        f.write("%s\n"%sys.argv[1])
        f.write("%s\n"%report)
        f.write("%s\n"%cm)
        f.write("---------------------\n")


def test2():
    print("Loading test data...")
    start_time = time.time()
    x_test, _, _ = process_file(
        val_dir, label_dir, word_to_id, cat_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1


    y_pred_probs = np.zeros(shape=[len(x_test), 4], dtype=np.float32)  # 保存预测结果
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        
        pred_cls, pred_probs = session.run(
            [model.y_pred_cls, model.y_pred_probs], feed_dict=feed_dict)

        y_pred_cls[start_id:end_id] = pred_cls
        y_pred_probs[start_id:end_id] = pred_probs

    # np.save(pred_path, y_pred_cls)

    #y
    data_df = pd.read_csv('data/raw/val/sentiment_analysis_validationset.csv', header=0, encoding='utf8')
    x_content = data_df['content']
    y_test = list(data_df[sys.argv[1]])
    y_test_cls = [cat_to_id[str(y)] for y in y_test]


    for i in range(num_classes):
        # class_name = class_name_list[i]
        class_pred_probs = y_pred_probs[:, i]
        analyse_fp = result_fp_list[i]

        sort_indices = np.argsort(class_pred_probs)[::-1]

        output_result_pd = pd.DataFrame(columns=['text', 'prob', 'label'])

        output_result_pd['text'] = [x_content[i] for i in sort_indices]
        output_result_pd['prob'] = [class_pred_probs[i] for i in sort_indices]
        output_result_pd['label'] = [y_test_cls[i] for i in sort_indices]

        output_result_pd.to_csv(analyse_fp, encoding='utf_8_sig', index=False)

    # 评估
    # print(sys.argv[1])
    # print("Precision, Recall and F1-Score...")
    # report = metrics.classification_report(
    #     y_test_cls, y_pred_cls, target_names=categories)
    # print(report)

    # # 混淆矩阵
    # print("Confusion Matrix...")
    # cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    # print(cm)

    # with open('log.txt', 'a') as f:
    #     f.write("%s\n" % sys.argv[1])
    #     f.write("%s\n" % report)
    #     f.write("%s\n" % cm)
    #     f.write("---------------------\n")


if __name__ == '__main__':
    config = TCNNConfig()
    
    config.loss_weight = [1 for c in range(4)]
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)

    model = TextCNN(config)
    test()

