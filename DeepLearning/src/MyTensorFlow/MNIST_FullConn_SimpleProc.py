# -*- coding: UTF-8 -*-

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from MNIST.GetDataSetClass import *
from CostFunc.CrossEntropyClass import *
from NormalizeInput.NormorlizeInputClass import *

file1 = 'E:/eclipse/eclipse-workspace/MNIST/train-images-idx3-ubyte'
file2 = 'E:/eclipse/eclipse-workspace/MNIST/train-labels-idx1-ubyte'
file3 = 'E:/eclipse/eclipse-workspace/MNIST/t10k-images-idx3-ubyte'
file4 = 'E:/eclipse/eclipse-workspace/MNIST/t10k-labels-idx1-ubyte'

s = 784
m = 10
x, head1 = loadImageSet(file1, m)
y, head2 = loadLabelSet(file2, m)

TrainData = x.T
TrainLabel = y.T

X = tf.placeholder(tf.float32, shape=[s, m])
Y = tf.placeholder(tf.float32, shape=[10, m])

n = [20, 10]

np.random.seed(1)
W1_0 = 0.1 * (np.random.randn(n[0],s))
b1_0 = np.zeros((n[0], 1))
W2_0 = 0.1 * (np.random.randn(n[1],n[0]))
b2_0 = np.zeros((n[1], 1))

W1 = tf.Variable(W1_0, dtype=tf.float32)
b1 = tf.Variable(b1_0, dtype=tf.float32)
W2 = tf.Variable(W2_0, dtype=tf.float32)
b2 = tf.Variable(b2_0, dtype=tf.float32)

Z1 = tf.matmul(W1,X) + b1
A1 = tf.sigmoid(Z1) 
Z2 = tf.matmul(W2,A1) + b2
A2 = tf.sigmoid(Z2)

rel = tf.argmax(A2,0)
Lable = tf.argmax(Y,0)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
 
Lable_list = sess.run(Lable, feed_dict={Y: TrainLabel})
print('Label:',Lable_list) 

L_mat = (-1) * (Y * tf.log(A2) + (1-Y) * tf.log(1-A2))
CostF = tf.reduce_sum(L_mat, 1, keep_dims=True) / m
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(CostF)

for i in range(5000):
    sess.run(train_step, feed_dict={X: TrainData, Y: TrainLabel}) 

Predict_list = sess.run(rel, feed_dict={X: TrainData})
print('Predict:',Predict_list)

correct_prediction = tf.equal(Predict_list, Lable_list)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('accuracy accuracy',sess.run(accuracy)) 