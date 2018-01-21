# -*- coding: UTF-8 -*-
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from MyTensorFlow.TF_CnnClass import *
from MyTensorFlow.TF_CostFuncClass import *
from MyTensorFlow.TF_FullConnClass import *


#x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
#x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
InputImgSize = 8
x = np.reshape(x, [-1,InputImgSize,InputImgSize,1])
m = x.shape[0]
print('m:', m)

X = tf.placeholder(tf.float32, shape=[m,8,8,1])

ConvFilterSet = [2,2]
ConvCH = [1,1]
ConvStridesSet = [1,1]
ConvPadSet = ['SAME','SAME']
PoolKsizeSet = [2,2]
PoolStridesSet = [2,2]
PoolPadSet = ['SAME','SAME']
act = tf.nn.relu
Activator = [act, act]
CP_Layer = ConvPoolNetwork(1, ConvFilterSet, ConvCH, ConvStridesSet, ConvPadSet, PoolKsizeSet, PoolStridesSet, PoolPadSet, Activator)
CP_A =  CP_Layer.ForwardPropagation(X)

ImgSizeAfterCP = SizeAfterCPNw(InputImgSize, ConvFilterSet, ConvStridesSet, PoolKsizeSet, PoolStridesSet)
FC_s = np.square(ImgSizeAfterCP)
print('FC_s:', FC_s)

'''
把卷基层输出的结果一维化
'''
CP_A_SHAPE = tf.shape(CP_A)
CP_A_FLAT_LEN = CP_A_SHAPE[1] * CP_A_SHAPE[2] * CP_A_SHAPE[3]
CP_A_FLAT = tf.reshape(CP_A, [-1, CP_A_FLAT_LEN])

'''
把一维化的结果列化，即每一列数据对应一个样本
'''
FC_INPUT = tf.transpose(CP_A_FLAT)
n = [10]
learn_rate = 0.1
epoch = 80
FC_Activator = tf.sigmoid


FC_A = TF_FullConnForwardPropagation(FC_s, n, FC_INPUT, FC_Activator)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

y = sess.run(FC_A, feed_dict={X: x})
print(y.shape)
#print(y)

sess.close()


'''
m=1
(1, 2, 2, 1)
[[[[ 0.        ]
   [ 0.76567018]]

  [[ 1.45644879]
   [ 2.69895244]]]]
'''
