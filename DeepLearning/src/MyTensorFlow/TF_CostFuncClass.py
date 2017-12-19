# -*- coding: UTF-8 -*-

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def TF_CrossEntropy(A, Y, m):
    L_mat = (-1) * (Y * tf.log(A) + (1-Y) * tf.log(1-A))
    return tf.reduce_sum(L_mat, 1, keep_dims=True) / m