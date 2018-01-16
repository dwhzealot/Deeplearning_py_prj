# -*- coding: UTF-8 -*-

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def ConvPoolLayer(object):
    def __init__(self, ConvFilter, ConvStrides, ConvPadding, PoolKsize, PoolStrides, PoolPadding, Activator):
        assert(len(ConvFilter) == 4)
        assert(len(ConvStrides) == 4)
        assert(len(PoolKsize) == 4)
        assert(len(PoolStrides) == 4)        
        self.ConvFilter = ConvFilter
        self.ConvStrides = ConvStrides
        self.ConvPadding = ConvPadding
        W_init = (1 / ConvFilter[0]) * np.random.randn(ConvFilter[0],ConvFilter[1],ConvFilter[2],ConvFilter[3])
        self.W = tf.Variable(W_init, dtype=tf.float32)
        b_init = 0.1 * np.ones(ConvFilter[3])
        self.b = tf.Variable(b_init, dtype=tf.float32)

        self.PoolKsize = PoolKsize
        self.PoolStrides = PoolStrides
        self.PoolPadding = PoolPadding
        self.Act = Activator
        return
    def Calc(self, X):
        CONV = tf.nn.conv2d(X, self.W, strides = self.ConvStrides, padding = self.ConvPadding) + self.b
        CONV_A = self.Act(CONV)
        POOL = tf.nn.max_pool(CONV_A, ksize=self.PoolKsize, strides=self.PoolStrides, padding=self.PoolPadding)
        return POOL

def ConvPoolNetwork(object):
    def __init__(self, InputCH, ConvFilterSet, ConvCH, ConvStridesSet, ConvPadSet, 
                                PoolKsizeSet, PoolStridesSet, PoolPadSet, Activator):
        self.l_max = len(ConvFilterSet)
        self.layer = []
        ConvInputCH = InputCH
        for i in range(self.l_max):
            ConvFilter = [ConvFilterSet[i], ConvFilterSet[i],ConvInputCH,ConvCH[i]]
            ConvStrides = [1, ConvStridesSet[i], ConvStridesSet[i], 1]
            ConvPadding = ConvPadSet[i]
            PoolKsize = [1, PoolKsizeSet[i], PoolKsizeSet[i], 1]
            PoolStrides = [1, PoolStridesSet[i], PoolStridesSet[i], 1]
            PoolPadding = PoolPadSet[i]
            layer = ConvPoolLayer(ConvFilter, ConvStrides, ConvPadding,
                                  PoolKsize, PoolStrides, PoolPadding, Activator[i])
            self.layer.append(layer)
            ConvInputCH = ConvCH[i]
        return
    def ForwardPropagation(self, X):
        A = X
        for i in range(self.l_max):
            A = self.layer[i].calc(A)
        return A              









