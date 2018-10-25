# coding: utf-8
import os
import sys
import numpy as np 
import tensorflow as tf

class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 64  # 词向量维度
    seq_length = 600  # 序列长度
    num_classes = 4  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 5000  # 词汇表达小

    hidden_dim = 1024  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    regularization_kernel_weight = 0.0005
    regularization_bias_weight = 0.001
    learning_rate = 1e-3  # 学习率

    batch_size = 1024  # 每批训练大小
    num_epochs = 100  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("cnn"):
            # CNN layer
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            # global max pooling layer
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_probs = tf.nn.softmax(self.logits)
            self.y_pred_cls = tf.argmax(self.y_pred_probs, 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            weithed_err = tf.multiply(cross_entropy, self.config.loss_weight)

            self.entropy_loss = tf.reduce_mean(cross_entropy)
            self.regularization_loss = self.regularization_kernel * self.config.regularization_kernel_weight \
                                    + self.regularization_bias * self.config.regularization_bias_weight

            self.loss = self.entropy_loss + self.regularization_loss
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        # with tf.name_scope("accuracy"):
            # pass
            # 准确率
            # correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            # self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            # score = metrics.f1_score(tf.argmax(self.input_y, 1).eval(session = sess), self.y_pred_cls.eval(session = sess), average = "macro")

            # print(score)
            # self.acc = tf.Variable(score, dtype=tf.float32, name = "acc")
            
    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    @property
    def regularization_kernel_vars(self):
        return [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'kernel' in var.name and 'fc' in var.name]

    @property
    def regularization_bias_vars(self):
        return [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'bias' in var.name and 'fc' in var.name]

    @property
    def regularization_kernel(self):
        return tf.reduce_sum([tf.reduce_mean(var) for var in self.regularization_kernel_vars])

    @property
    def regularization_bias(self):
        return tf.reduce_sum([tf.reduce_mean(var) for var in self.regularization_bias_vars])


if __name__ == '__main__':

    from data.cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab

    base_dir = './data/new/'

    
    vocab_dir = os.path.join(base_dir, 'cnews.vocab_' + sys.argv[1] +'.txt')
    train_dir = os.path.join(base_dir, 'train_split_search.txt')
    test_dir = os.path.join(base_dir, 'val_split_search.txt')
    val_dir = os.path.join(base_dir, 'val_split_search.txt')

    train_label_dir = os.path.join(base_dir, 'train_' + sys.argv[1] +'.txt')
    val_label_dir = os.path.join(base_dir, 'val_' + sys.argv[1] + '.txt')
    test_label_dir = val_label_dir

    print('Configuring CNN model...')
    config = TCNNConfig()

    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, train_label_dir, vocab_dir, config.vocab_size)
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)

    x_train, y_train, train_label_weight = process_file(train_dir, train_label_dir, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val, val_label_weight = process_file(val_dir, val_label_dir, word_to_id, cat_to_id, config.seq_length)
    config.loss_weight = train_label_weight

    print('train_label_weight: ', train_label_weight)
    print('val_label_weight: ', val_label_weight)


    model = TextCNN(config)



    for var in model.trainable_vars:
        print(var.name, ' --> ', var.get_shape())