import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from MNIST.GetDataSetClass import *


#x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
x = np.reshape(x, [-1,4,4,1])
#print(x)

X = tf.placeholder(tf.float32, shape=[1,4,4,1])

W_init = np.ones((2,2,1,1))
TF_W = tf.Variable(W_init, dtype=tf.float32)
b_init = np.ones(1)
TF_b_1 = tf.Variable(b_init, dtype=tf.float32)

CONV = tf.nn.conv2d(X, TF_W, strides=[1, 1, 1, 1], padding='SAME') + TF_b_1
POOL = tf.nn.max_pool(CONV, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

np.random.seed(5)
W_init_3 = np.random.randn(4,10)
TF_W_3 = tf.Variable(W_init_3, dtype=tf.float32)
b_init_3 = 0.1 * np.ones(10)
TF_b_3 = tf.Variable(b_init_3, dtype=tf.float32)

POOL_flat = tf.reshape(POOL, [1, 4])
FC_3 = tf.matmul(POOL_flat, TF_W_3) + TF_b_3
A_FC_3 = tf.sigmoid(FC_3)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

conv_res = sess.run(CONV, feed_dict={X:x}) 
print('conv_res shape', conv_res.shape)

conv_res_img = np.reshape(conv_res, (4,4))
print('conv_res_img\n',conv_res_img)

pool_res = sess.run(POOL, feed_dict={X:x})
print('pool_res shape', pool_res.shape)


pool_res_img = np.reshape(pool_res, (2,2))
print('pool_res_img\n',pool_res_img)

fc_res = sess.run(A_FC_3, feed_dict={X:x})
print('pool_res shape', fc_res.shape)
print(fc_res)

sess.close()
'''
conv_res shape (1, 4, 4, 1)
conv_res_img
 [[ 15.  19.  23.  13.]
 [ 31.  35.  39.  21.]
 [ 47.  51.  55.  29.]
 [ 28.  30.  32.  17.]]
pool_res shape (1, 2, 2, 1)
pool_res_img
 [[ 35.  39.]
 [ 51.  55.]]
pool_res shape (1, 10)
[[  5.90627405e-17   0.00000000e+00   1.00000000e+00   4.57702617e-06
    2.15240902e-06   1.00000000e+00   1.00000000e+00   1.00000000e+00
    1.98240265e-25   1.00000000e+00]]
    
'''


