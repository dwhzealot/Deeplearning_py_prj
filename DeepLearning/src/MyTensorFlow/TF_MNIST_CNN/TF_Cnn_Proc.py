# -*- coding: UTF-8 -*-

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from MNIST.GetDataSetClass import *
from MyTensorFlow.TF_CostFuncClass import *

print('TF_MNIST_CNN_test start')
m = 512
print('Getting Trainset...')
TrainSet = MNIST_getDataSet(MNIST_TrainDataSet, MNIST_TrainLabelSet)
TrainDataSet, TrainLabelSet, minibatch_num = TrainSet.minibatch(m, Nmlz = True, Train = True)
#TrainDataSet, TrainLabelSet, minibatch_num = TrainSet.minibatch(m)

X = tf.placeholder(tf.float32, shape=[m,28,28,1])
Y = tf.placeholder(tf.float32, shape=[10, None])
'''
第1个卷积池化层
'''
np.random.seed(1)
W_init_1 = 0.1 * np.random.randn(5,5,1,32)
TF_W_1 = tf.Variable(W_init_1, dtype=tf.float32)
b_init_1 = 0.1 * np.ones(32)
TF_b_1 = tf.Variable(b_init_1, dtype=tf.float32)

CONV_1 = tf.nn.conv2d(X, TF_W_1, strides=[1, 1, 1, 1], padding='SAME') + TF_b_1
A_CONV_1 = tf.nn.relu(CONV_1)
POOL_1 = tf.nn.max_pool(A_CONV_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
'''
第2个卷积池化层
'''

W_init_2 = 0.1 * np.random.randn(5,5,32,64)
TF_W_2 = tf.Variable(W_init_2, dtype=tf.float32)
b_init_2 = 0.1 * np.ones(64)
TF_b_2 = tf.Variable(b_init_2, dtype=tf.float32)

CONV_2 = tf.nn.conv2d(POOL_1, TF_W_2, strides=[1, 1, 1, 1], padding='SAME') + TF_b_2
A_CONV_2 = tf.nn.relu(CONV_2)
POOL_2 = tf.nn.max_pool(A_CONV_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

'''
第3层，全连接层
'''
POOL_2_flat = tf.reshape(POOL_2, [-1, 7*7*64])
POOL_2_flat_T = tf.transpose(POOL_2_flat)

W_init_3 = 0.1 * np.random.randn(100, 7*7*64)
TF_W_3 = tf.Variable(W_init_3, dtype=tf.float32)
b_init_3 = 0.1 * np.ones((100,1))
TF_b_3 = tf.Variable(b_init_3, dtype=tf.float32)

FC_3 = tf.matmul(TF_W_3, POOL_2_flat_T) + TF_b_3
A_FC_3 = tf.sigmoid(FC_3)

'''
第4层，输出层，softmax层
'''
W_init_4 = 0.1 * np.random.randn(10,100)
TF_W_4 = tf.Variable(W_init_4, dtype=tf.float32)
b_init_4 = 0.1 * np.ones((10,1))
TF_b_4 = tf.Variable(b_init_4, dtype=tf.float32)

FC_4 = tf.matmul(TF_W_4, A_FC_3) + TF_b_4
A_FC_4 = tf.sigmoid(FC_4)

Lable = tf.argmax(Y,0)
rel = tf.argmax(A_FC_4,0)

'''
计算成本函数
'''
CostF = TF_LogisticRegressionCrossEntropy(A_FC_4, Y, m)
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(CostF)
#train_step = tf.train.AdamOptimizer(0.002, 0.9, 0.999).minimize(CostF)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('Start training')
'''
for i in range(50):
    sess.run(train_step, feed_dict={X: x_image, Y: TrainLabelSet[0]}) 
'''
epoch = 1
cycle = minibatch_num-1
for i in range(epoch):
    '''
    if i % (epoch // 10) == 0:
        progress = i / epoch *100
        print('Training progress: ', progress,'%')
    if i == (epoch-1):
        print('Training progress: 100%')
    '''
    for j in range(cycle): 
        '''
        #由于最后一个minibatch的size与其他的minibatch的size可能不同.
        #在Cost函数的公式中，由于TF的占位符规则，m无法动态变化，就无法兼容最后一个minibatch的Cost计算
        #所以训练时，不训练最后一个minibatch
        '''        
        if j % (cycle // 10) == 0:
            progress = j *100 // cycle 
            print('Training progress: ', progress,'%')
        if j == (cycle - 1):
            print('Training progress: 100%')
        x_image = np.reshape(TrainDataSet[j].T, [-1,28,28,1])
        sess.run(train_step, feed_dict={X: x_image, Y: TrainLabelSet[j]}) 


Train_Lable_list = sess.run(Lable, feed_dict={Y: TrainLabelSet[0]})
#print('label  ', Train_Lable_list)
train_test = np.reshape(TrainDataSet[0].T, [-1,28,28,1])
Train_Predict_list = sess.run(rel, feed_dict={X: train_test})
#print('predict', Train_Predict_list)
correct_prediction = tf.equal(Train_Predict_list, Train_Lable_list)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Trainset predict accuracy',sess.run(accuracy))


print('Getting Testset...')
Test_mean = TrainSet.NmlzMean
Test_variance = TrainSet.NmlzVariance
TestSet = MNIST_getDataSet(MNIST_TestDataSet, MNIST_TestLabelSet)
TestDataSet, TestLabelSet, test_minibatch_num =TestSet.minibatch(m, Nmlz = True, Train = False,mean = Test_mean, variance=Test_variance)
#TestDataSet, TestLabelSet, test_minibatch_num =TestSet.minibatch(m)

print('Start testing Testset')
Lable_list = sess.run(Lable, feed_dict={Y: TestLabelSet[0]})
test_test = np.reshape(TestDataSet[0].T, [-1,28,28,1])
Predict_list = sess.run(rel, feed_dict={X: test_test})
correct_prediction = tf.equal(Predict_list, Lable_list)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Testset predict accuracy',sess.run(accuracy)) 


sess.close()

