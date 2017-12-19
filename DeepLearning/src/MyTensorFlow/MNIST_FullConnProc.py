# -*- coding: UTF-8 -*-

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from MNIST.GetDataSetClass import *
from WeightInit.WeightInitClass import *
from TF_CostFuncClass import *

minibatch_size = 512
test_minibatch_size = 512
TrainSet = MNIST_getDataSet(MNIST_TrainDataSet, MNIST_TrainLabelSet)
TrainDataSet, TrainLabelSet, minibatch_num = TrainSet.minibatch(minibatch_size)
print('Train minibatch_num',minibatch_num)
TestSet = MNIST_getDataSet(MNIST_TestDataSet, MNIST_TestLabelSet)
TestDataSet, TestLabelSet, test_minibatch_num = TestSet.minibatch(minibatch_size)
print('Test minibatch_num',test_minibatch_num)

s = 784
m = minibatch_size
n = [40, 30, 20, 10]
learn_rate = 0.1

X = tf.placeholder(tf.float32, shape=[s, None])
Y = tf.placeholder(tf.float32, shape=[10, None])
l_max = len(n)
A = 0

for i in range(l_max):
    if i == 0:
        column = s
        input = X
    else:
        column = n[i-1]
        input = A
    np.random.seed(n[i])
    W_ini_coe = Sigmoid_W_init_coe(column)
    W_init = W_ini_coe * (np.random.randn(n[i],column))
    W_var = tf.Variable(W_init, dtype=tf.float32)
    b_init = np.zeros((n[i], 1))
    b_var = tf.Variable(b_init, dtype=tf.float32)
    
    Z = tf.matmul(W_var,input) + b_var
    A = tf.sigmoid(Z)
    
Lable = tf.argmax(Y,0)
rel = tf.argmax(A,0)

CostF = TF_LogisticRegressionCrossEntropy(A, Y, m)

#train_step = tf.train.AdamOptimizer(0.01, 0.9, 0.999).minimize(CostF)
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(CostF)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(50):
    if i%10 == 0:
        print('epoch',i)
    
    for j in range(minibatch_num):
        sess.run(train_step, feed_dict={X: TrainDataSet[j], Y: TrainLabelSet[j]}) 

Lable_list = sess.run(Lable, feed_dict={Y: TestLabelSet[17]})
Predict_list = sess.run(rel, feed_dict={X: TestDataSet[17]})

correct_prediction = tf.equal(Predict_list, Lable_list)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('accuracy',sess.run(accuracy)) 
