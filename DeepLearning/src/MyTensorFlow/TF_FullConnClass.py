# -*- coding: UTF-8 -*-

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from WeightInit.WeightInitClass import *

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