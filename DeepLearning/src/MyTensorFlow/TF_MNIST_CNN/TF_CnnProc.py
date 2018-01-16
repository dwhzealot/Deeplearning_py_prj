import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from MyTensorFlow.TF_CnnClass import *

#x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
#x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
x = np.reshape(x, [-1,8,8,1])
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
A =  CP_Layer.ForwardPropagation(X)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

y = sess.run(A, feed_dict={X: x})
print(y.shape)
print(y)

sess.close()


'''
m=1
(1, 2, 2, 1)
[[[[ 0.        ]
   [ 0.76567018]]

  [[ 1.45644879]
   [ 2.69895244]]]]
'''
