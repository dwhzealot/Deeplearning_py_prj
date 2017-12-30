# -*- coding: UTF-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def TF_LogisticRegressionCrossEntropy(A, Y, m):
    L_mat = (-1) * (Y * tf.log(A) + (1-Y) * tf.log(1-A))
    return tf.reduce_sum(L_mat, 1, keep_dims=True) / m

def TF_LogisticRegressionCrossEntropy_2(A, Y, m):
    L_mat = (-1) * (Y * tf.log(A) + (1-Y) * tf.log(1-A))
    cost_mat =  tf.reduce_sum(L_mat, 1, keep_dims=True) / m
    return tf.reduce_sum(cost_mat, 0, keep_dims=True) / 10

def TF_LogisticRegressionCrossEntropy_L2Regularize(A, Y, m, CollectionName):
    L_mat = (-1) * (Y * tf.log(A) + (1-Y) * tf.log(1-A))
    cost_mat =  tf.reduce_sum(L_mat, 1, keep_dims=False) / m
    cost = tf.reduce_sum(cost_mat, 0, keep_dims=False) / 10
    tf.add_to_collection(CollectionName, cost)
    return tf.add_n(tf.get_collection(CollectionName))


def TF_SoftmaxRegressionCrossEntropy(A, Y, m):
    L_mat = (-1) * tf.reduce_sum((Y * tf.log(A)), 0,  keep_dims=True)
    return tf.reduce_sum(L_mat, 1, keep_dims=True) / m

def TF_SoftmaxRegressionCrossEntropy_L2Regularize(A, Y, m, CollectionName):
    L_mat = (-1) * tf.reduce_sum((Y * tf.log(A)), 0,  keep_dims=False)
    cost = tf.reduce_sum(L_mat, 1) / m
    tf.add_to_collection(CollectionName, cost)
    return tf.add_n(tf.get_collection(CollectionName))
