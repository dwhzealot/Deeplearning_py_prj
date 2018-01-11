import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from MNIST.GetDataSetClass import *
from MyTensorFlow.TF_CnnClass import *

file1 = 'E:/eclipse/eclipse-workspace/MNIST/train-images-idx3-ubyte'
file2 = 'E:/eclipse/eclipse-workspace/MNIST/train-labels-idx1-ubyte'
file3 = 'E:/eclipse/eclipse-workspace/MNIST/t10k-images-idx3-ubyte'
file4 = 'E:/eclipse/eclipse-workspace/MNIST/t10k-labels-idx1-ubyte'

s = 784
m = 1
x, head1 = loadImageSet(file1, m)
y, head2 = loadLabelSet(file2, m)

x_image = np.reshape(x, [-1,28,28,1])


X = tf.placeholder(tf.float32, shape=[m,28,28,1])
'''
第1个卷积池化层
'''
W_init_1 = np.random.randn(5,5,1,32)
TF_W_1 = tf.Variable(W_init_1, dtype=tf.float32)
b_init_1 = 0.1 * np.ones(32)
TF_b_1 = tf.Variable(b_init_1, dtype=tf.float32)

CONV_1 = tf.nn.conv2d(X, TF_W_1, strides=[1, 1, 1, 1], padding='SAME') + TF_b_1
A_CONV_1 = tf.nn.relu(CONV_1)
POOL_1 = tf.nn.max_pool(A_CONV_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
'''
第2个卷积池化层
'''
W_init_2 = np.random.randn(5,5,32,64)
TF_W_2 = tf.Variable(W_init_2, dtype=tf.float32)
b_init_2 = 0.1 * np.ones(64)
TF_b_2 = tf.Variable(b_init_2, dtype=tf.float32)

CONV_2 = tf.nn.conv2d(POOL_1, TF_W_2, strides=[1, 1, 1, 1], padding='SAME') + TF_b_2
A_CONV_2 = tf.nn.relu(CONV_2)
POOL_2 = tf.nn.max_pool(A_CONV_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

a_conv_res_1 = sess.run(A_CONV_1, feed_dict={X:x_image}) 
print('a_conv_res_1 shape', a_conv_res_1.shape)
pool_res_1 = sess.run(POOL_1, feed_dict={X:x_image})
print('pool_res_1 shape', pool_res_1.shape)

a_conv_res_2 = sess.run(A_CONV_2, feed_dict={X:x_image}) 
print('a_conv_res_2 shape', a_conv_res_2.shape)
pool_res_2 = sess.run(POOL_2, feed_dict={X:x_image})
print('pool_res_1 shape', pool_res_2.shape)

sess.close()
'''
a_conv_res_1 shape (1, 28, 28, 32)
pool_res_1 shape (1, 14, 14, 32)
a_conv_res_2 shape (1, 14, 14, 64)
pool_res_1 shape (1, 7, 7, 64)

'''