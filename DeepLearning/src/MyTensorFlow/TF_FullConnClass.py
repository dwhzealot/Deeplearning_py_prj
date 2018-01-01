# -*- coding: UTF-8 -*-

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from WeightInit.WeightInitClass import *
from MyTensorFlow.TF_RegularizeClass import *
from MyTensorFlow.TF_NormalizeClass import *
from tensorflow.python.training.moving_averages import assign_moving_average

def TF_FullConnForwardPropagation(s, n, X, Activator):
    l_max = len(n)
    A = 0
    
    for i in range(l_max):
        if i == 0:
            column = s
            l_input = X
        else:
            column = n[i-1]
            l_input = A
        np.random.seed(n[i])
        W_ini_coe = Sigmoid_W_init_coe(column)
        W_init = W_ini_coe * (np.random.randn(n[i],column))
        W_var = tf.Variable(W_init, dtype=tf.float32)
        b_init = np.zeros((n[i], 1))
        b_var = tf.Variable(b_init, dtype=tf.float32)
        
        Z = tf.matmul(W_var,l_input) + b_var
        A = Activator(Z)
    return A

def TF_FullConnForwardPropagation_L2Regularize(s, n, X, Activator, lamda, CollectionName):
    l_max = len(n)
    A = 0
    
    for i in range(l_max):
        if i == 0:
            column = s
            l_input = X
        else:
            column = n[i-1]
            l_input = A
        np.random.seed(n[i])
        W_ini_coe = Sigmoid_W_init_coe(column)
        W_init = W_ini_coe * (np.random.randn(n[i],column))        
        W_var = tf.Variable(W_init, dtype=tf.float32)
        TF_L2RegularizeWeight(W_var, lamda, CollectionName)
        b_init = np.zeros((n[i], 1))
        b_var = tf.Variable(b_init, dtype=tf.float32)
        
        Z = tf.matmul(W_var,l_input) + b_var
        A = Activator(Z)
    return A

def TF_FullConnForwardPropagation_BatchNorm(s, n, X, Activator):
    l_max = len(n)
    A = 0
    
    for i in range(l_max):
        if i == 0:
            column = s
            l_input = X
        else:
            column = n[i-1]
            l_input = A
        np.random.seed(n[i])
        W_ini_coe = Sigmoid_W_init_coe(column)
        W_init = W_ini_coe * (np.random.randn(n[i],column))
        W_var = tf.Variable(W_init, dtype=tf.float32)       
        Z = tf.matmul(W_var,l_input)
        
        # 计算均值和方差，axes参数0表示batch维度          
        fc_mean, fc_var = tf.nn.moments(Z, axes= 1, keep_dims=True) 
        scale = tf.Variable(tf.ones([n[i],1]))  
        shift = tf.Variable(tf.zeros([n[i],1]))    
        epsilon = 1e-05  
        
        # 定义滑动平均模型对象  
        ema = tf.train.ExponentialMovingAverage(decay=0.5)  
        
        def mean_var_with_update():  
            ema_apply_op = ema.apply([fc_mean, fc_var])  
            with tf.control_dependencies([ema_apply_op]):  
                return tf.identity(fc_mean), tf.identity(fc_var)  
        mean, var = mean_var_with_update()
        Z_norm = tf.nn.batch_normalization(Z, mean, var, shift, scale, epsilon)                                                
        A = Activator(Z_norm)
    return A






