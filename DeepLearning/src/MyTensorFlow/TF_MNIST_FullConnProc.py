# -*- coding: UTF-8 -*-

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from MNIST.GetDataSetClass import *
from MyTensorFlow.TF_CostFuncClass import *
from MyTensorFlow.TF_FullConnClass import *

print('Traing start')
minibatch_size = 512
test_minibatch_size = 10000
s = 784
n = [40, 30, 20, 10]
learn_rate = 0.1
epoch = 50
Activator = tf.sigmoid

TrainSet = MNIST_getDataSet(MNIST_TrainDataSet, MNIST_TrainLabelSet)
TrainDataSet, TrainLabelSet, minibatch_num = TrainSet.minibatch(minibatch_size)
#print('Train minibatch_num',minibatch_num)
X = tf.placeholder(tf.float32, shape=[s, None])
Y = tf.placeholder(tf.float32, shape=[10, None])

A = TF_FullConnForwardPropagation(s, n, X, Activator)

CostF = TF_LogisticRegressionCrossEntropy(A, Y, minibatch_size)

#train_step = tf.train.AdamOptimizer(0.01, 0.9, 0.999).minimize(CostF)
train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(CostF)

Lable = tf.argmax(Y,0)
rel = tf.argmax(A,0)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(epoch):
    if i % (epoch // 10) == 0:
        progress = i / epoch *100
        print('Training progress: ', progress,'%')
    if i == (epoch-1):
        print('Training progress: 100%')

    for j in range(minibatch_num-1): 
        '''
        #由于最后一个minibatch的size与其他的minibatch的size可能不同.
        #在Cost函数的公式中，由于TF的占位符规则，m无法动态变化，就无法兼容最后一个minibatch的Cost计算
        #所以训练时，不训练最后一个minibatch
        '''
        sess.run(train_step, feed_dict={X: TrainDataSet[j], Y: TrainLabelSet[j]}) 

print('Test start, sample number: ',test_minibatch_size)

TrainSet_test, TrainSet_label, traintest_minibatch_num = TrainSet.minibatch(test_minibatch_size)
Train_Lable_list = sess.run(Lable, feed_dict={Y: TrainSet_label[3]})
Train_Predict_list = sess.run(rel, feed_dict={X: TrainSet_test[3]})
correct_prediction = tf.equal(Train_Predict_list, Train_Lable_list)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Trainset accuracy',sess.run(accuracy))


TestSet = MNIST_getDataSet(MNIST_TestDataSet, MNIST_TestLabelSet)
TestDataSet, TestLabelSet, test_minibatch_num =TestSet.minibatch(test_minibatch_size)
Lable_list = sess.run(Lable, feed_dict={Y: TestLabelSet[0]})
Predict_list = sess.run(rel, feed_dict={X: TestDataSet[0]})
correct_prediction = tf.equal(Predict_list, Lable_list)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Testset accuracy',sess.run(accuracy)) 
sess.close()