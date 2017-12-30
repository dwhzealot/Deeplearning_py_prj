# -*- coding: UTF-8 -*-

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average
from Normalize.NormorlizeClass import *


def batch_norm(x, train, eps=1e-05, decay=0.9, affine=True, name=None):
    with tf.variable_scope(name, default_name='BatchNorm2d'):
        params_shape = tf.shape(x)[-1:]
        moving_mean = tf.get_variable('mean', params_shape,
                                      initializer=tf.zeros_initializer,
                                      trainable=False)
        moving_variance = tf.get_variable('variance', params_shape,
                                          initializer=tf.ones_initializer,
                                          trainable=False)

        def mean_var_with_update():
            mean, variance = tf.nn.moments(x, tf.shape(x)[:-1], name='moments')
            with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                          assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(mean), tf.identity(variance)
        mean, variance = tf.cond(train, mean_var_with_update, lambda: (moving_mean, moving_variance))
        if affine:
            beta = tf.get_variable('beta', params_shape,
                                   initializer=tf.zeros_initializer)
            gamma = tf.get_variable('gamma', params_shape,
                                    initializer=tf.ones_initializer)
            x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
        else:
            x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
        return x
    
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
