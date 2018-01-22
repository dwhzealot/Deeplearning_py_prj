# -*- coding: UTF-8 -*-

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def TF_Sigmoid_FC_W_init_coe(n):
    '''
    n : W的列数，即W的特征个数
    '''
    a = tf.divide(1,n)
    return tf.sqrt(a)
