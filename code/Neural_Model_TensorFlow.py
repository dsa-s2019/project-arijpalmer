#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 01:37:33 2019

@author: ari
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import roc_auc_score

tf.reset_default_graph()

data = pd.read_csv('train_clean_pp_I1.csv')
data['year'].values[data['year'] == 2017] = 2016
#data.loc[data['tenure'] >=0, 'tenure'] = 1.
#data.loc[data['tenure'] < 0, 'tenure'] = 0.
data_train = data[data.train == 1]
data_train = data_train.drop(['state','train','id','credit_o'], axis=1)

data_test = data[data.train == 0]
data_test = data_test.drop(['state','train','id','cancel_1.0','credit_o'], axis=1)
#data['kar'] = data.nkids/data.nadults

y_dat = data_train['cancel_1.0']
X_dat = np.asarray(data_train.drop(['cancel_1.0'], axis=1))

y_dat = np.asarray(y_dat)
enc = OneHotEncoder()
enc.fit(y_dat.reshape(-1,1))
out = enc.transform(y_dat.reshape(-1,1)).toarray()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_dat, out, test_size=0.1, random_state=0)

n_inputs = 33  
n_hidden1 = 50
n_hidden2 = 10
n_hidden3 = 10
n_outputs = 2

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch
        
tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
y = tf.placeholder(tf.int32, shape=[None,2], name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    #hidden3 = tf.layers.dense(hidden2, n_hidden2, name="hidden3", activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
    y_proba = tf.nn.softmax(logits)
    
with tf.name_scope("loss"):
    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    
#learning_rate = 0.0

with tf.name_scope("train"):
    optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-5)
    training_op = optimizer.minimize(loss)
    
with tf.name_scope("eval"):
    labels = tf.argmax(y, 1)
    correct = tf.math.in_top_k(logits, labels, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    


init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 200
batch_size = 500

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_valid = accuracy.eval(feed_dict={X: X_test, y: y_test})
        brt = y_proba.eval(feed_dict={X: X_test})
        fpr2, tpr2, thresholds = metrics.roc_curve(y_test[:,1], brt[:,1], pos_label=1)
        auc2 = metrics.auc(fpr2, tpr2)
        print(epoch, "Batch accuracy:", acc_batch, "Validation accuracy:", acc_valid, "AUC:", auc2)

    #save_path = saver.save(sess, "./my_model_final.ckpt")

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC1 = %0.8f' % auc)
#plt.plot(fpr2, tpr2, 'k', label = 'AUC2 = %0.8f' % roc_auc2)
#plt.plot(fpr3, tpr3, 'g', label = 'AUC3 = %0.8f' % roc_auc3)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#plt.savefig('auc.png', bbox_inches='tight')
    