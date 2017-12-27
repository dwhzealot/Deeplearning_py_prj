# -*- coding: UTF-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def TF_L2RegularizeWeight(tf_W, lamda, CollectionName):
    tf.add_to_collection(CollectionName, tf.contrib.layers.l2_regularizer(lamda)(tf_W))
    return
