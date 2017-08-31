import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import shutil
from libs import (get_variable, get_conv, get_bias, get_pool, conv_and_pool)



with open('data/image_', 'rb') as f:
    image = pickle.load(f)

with open('data/ncratio100_', 'rb') as f:
    ncratio = pickle.load(f)



class CNN:
    def __init__(self):
        with tf.Graph().as_default():
            self.prepare_model()
            self.prepare_session()

    def prepare_model(self):
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, 360 * 360])
            x_image = tf.reshape(x, [-1,360,360,1])

        with tf.name_scope('conv_and_pool1'):
            num_filters1 = 32
            h_conv1, h_pool1 = conv_and_pool(x_image, 1, num_filters1, 10, 2)

        with tf.name_scope('conv_and_pool2'):
            num_filters2 = 64
            h_conv2, h_pool2 = conv_and_pool(h_pool1, num_filters1, num_filters2, 5, 1)

        with tf.name_scope('conv_and_pool3'):
            num_filters3 = 64
            h_conv3, h_pool3 = conv_and_pool(h_pool2, num_filters2, num_filters3, 3, 1)

        with tf.name_scope('conv_and_pool4'):
            num_filters4 = 64
            h_conv4, h_pool4 = conv_and_pool(h_pool3, num_filters3, num_filters4, 3, 1)

        with tf.name_scope('fully_connected'):
            final_pixel = 9
            h_pool_flat = tf.reshape(h_pool4, [-1, (final_pixel**2)*num_filters4])

            num_units1 = (final_pixel**2)*num_filters4
            num_units2 = 1024

            w2 = get_variable([num_units1, num_units2])
            b2 = get_bias([num_units2])
            hidden2 = tf.nn.relu(tf.matmul(h_pool_flat, w2) + b2)

        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            hidden2_drop = tf.nn.dropout(hidden2, keep_prob)

        with tf.name_scope('softmax'):
            num_class = 100

            w0 = get_variable([num_units2, num_class])
            b0 = get_bias([num_class])
            p = tf.nn.softmax(tf.matmul(hidden2_drop, w0) + b0)

        with tf.name_scope('optimizer'):
            t = tf.placeholder(tf.float32, [None, num_class])
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t,logits=p))
            train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
            correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.name_scope('evaluator'):
            correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.histogram("convolution_filters1", h_conv1)
        tf.summary.histogram("convolution_filters2", h_conv2)
        
        self.x, self.t, self.p, self.keep_prob = x, t, p, keep_prob
        self.train_step = train_step
        self.loss = loss
        self.accuracy = accuracy

    def prepare_session(self):
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        summary = tf.summary.merge_all()

        saver = tf.train.Saver()
        if os.path.isdir('/tmp/logs'):
            shutil.rmtree('/tmp/logs')
        writer = tf.summary.FileWriter("/tmp/logs", sess.graph)
        
        self.sess = sess
        self.summary = summary
        self.writer = writer
        self.saver = saver

cnn = CNN()

i = 0
for _ in range(100):
    i += 1
    cnn.sess.run(cnn.train_step,
             feed_dict={cnn.x:image, cnn.t:ncratio, cnn.keep_prob:0.1})
    if i % 1 == 0:
        summary, loss_val, acc_val = cnn.sess.run([cnn.summary, cnn.loss, cnn.accuracy],
                feed_dict={cnn.x:image,
                           cnn.t:ncratio,
                           cnn.keep_prob:1.0})
        print ('Step: %d, Loss: %f, Accuracy: %f'
               % (i, loss_val, acc_val))
        # cnn.saver.save(cnn.sess, os.path.join(os.getcwd(), 'cnn_session'), global_step=i)
        cnn.writer.add_summary(summary, i)
