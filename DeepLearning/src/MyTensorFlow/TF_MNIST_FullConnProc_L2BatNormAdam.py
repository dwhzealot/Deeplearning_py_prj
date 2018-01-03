# -*- coding: UTF-8 -*-
# -*- coding: UTF-8 -*-

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from MNIST.GetDataSetClass import *
from MyTensorFlow.TF_CostFuncClass import *
from MyTensorFlow.TF_FullConnClass import *

print('TF_MNIST_FullConnL2RegularizeProc Traing start')
minibatch_size = 512
test_minibatch_size = 10000
s = 784
n = [40, 30, 20, 10]
learn_rate = 0.002
epoch = 500
Activator = tf.sigmoid
CollectionName = 'loss'
lamda = 0.001
print('Getting Trainset...')
TrainSet = MNIST_getDataSet(MNIST_TrainDataSet, MNIST_TrainLabelSet)
TrainDataSet, TrainLabelSet, minibatch_num = TrainSet.minibatch(minibatch_size)

X = tf.placeholder(tf.float32, shape=[s, None])
Y = tf.placeholder(tf.float32, shape=[10, None])

A = TF_FullConnForwardPropagation_L2Regularize(s, n, X, Activator, lamda, CollectionName)

CostF = TF_LogisticRegressionCrossEntropy_L2Regularize(A, Y, minibatch_size, CollectionName)

train_step = tf.train.AdamOptimizer(learn_rate, 0.9, 0.999).minimize(CostF)
#train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(CostF)

Lable = tf.argmax(Y,0)
rel = tf.argmax(A,0)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('Start training')
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

print('Start testing Trainset')
'''
TrainSet中的数据已经在训练时被归一化，此处无须再次归一化，仅重新设置minibatchSize
'''
TrainSet_test, TrainSet_label, traintest_minibatch_num = TrainSet.minibatch(test_minibatch_size)
Train_Lable_list = sess.run(Lable, feed_dict={Y: TrainSet_label[3]})
Train_Predict_list = sess.run(rel, feed_dict={X: TrainSet_test[3]})
correct_prediction = tf.equal(Train_Predict_list, Train_Lable_list)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Trainset predict accuracy',sess.run(accuracy))

print('Getting Testset...')
Test_mean = TrainSet.NmlzMean
Test_variance = TrainSet.NmlzVariance
TestSet = MNIST_getDataSet(MNIST_TestDataSet, MNIST_TestLabelSet)
TestDataSet, TestLabelSet, test_minibatch_num =TestSet.minibatch(test_minibatch_size)

print('Start testing Testset')
Lable_list = sess.run(Lable, feed_dict={Y: TestLabelSet[0]})
Predict_list = sess.run(rel, feed_dict={X: TestDataSet[0]})
correct_prediction = tf.equal(Predict_list, Lable_list)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Testset predict accuracy',sess.run(accuracy)) 
sess.close()