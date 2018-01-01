# -*- coding: UTF-8 -*-

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average
from Normalize.NormorlizeClass import *
  
def TF_GetMeanVariance(InputMat):
    TF_Input = tf.placeholder(tf.float32, shape=[None, None])
    TF_mean, TF_variance = tf.nn.moments(TF_Input, axes = 1, keep_dims=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run([TF_mean, TF_variance], feed_dict = {TF_Input: InputMat})

def TF_NormalizeData(InputRow, mean, variance):
    eps = 1e-05
    TF_InputRow = tf.placeholder(tf.float32, shape=[None, None])
    Nmlz = tf.nn.batch_normalization(TF_InputRow, mean, variance, None, None, eps)
    InputNmlz = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run(Nmlz, feed_dict={TF_InputRow: InputRow})

'''
x = [1,2,3,4,5,6,7,8,9,10,11,12]
X = np.reshape(x, [3,4])
print(X)
mean , vari = TF_GetMeanVariance(X)
print(mean)
print(vari)
print(TF_NormalizeData(X, mean, vari))

nml_np = Normalize()
X_np = nml_np.ProcTrainData(X)
print(nml_np.miu)
print(nml_np.sigmaSquare)
print(X_np)
'''
