import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from MNIST.GetDataSetClass import *
from MyTensorFlow.TF_CnnClass import *

x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
x = np.reshape(x, [-1,4,4,1])
#print(x)

X = tf.placeholder(tf.float32, shape=[1,4,4,1])

W_init = np.ones((2,2,1,1))
TF_W = tf.Variable(W_init, dtype=tf.float32)

CONV = tf.nn.conv2d(X, TF_W, strides=[1, 1, 1, 1], padding='SAME')
POOL = tf.nn.max_pool(CONV, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

conv_res = sess.run(CONV, feed_dict={X:x}) 
print('conv_res shape', conv_res.shape)
conv_res_img = np.reshape(conv_res, (4,4))
print('conv_res_img\n',conv_res_img)

pool_res = sess.run(POOL, feed_dict={X:x})
print('pool_res shape', conv_res.shape)
pool_res_img = np.reshape(pool_res, (2,2))
print('pool_res_img\n',pool_res_img)

sess.close()
