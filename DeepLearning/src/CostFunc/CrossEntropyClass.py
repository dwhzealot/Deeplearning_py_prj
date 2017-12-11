# -*- coding: UTF-8 -*-
'''
Created on 2017年12月6日

@author: dongwh3
'''
import numpy as np
class CrossEntropy(object):
    def Calculate(self, A, Y):
        assert(A.shape[0] == 1)
        assert(Y.shape[0] == 1)
        assert(A.shape[1] == Y.shape[1])
        m = A.shape[1]
        L_mat = (-1) * (Y * np.log(A) + (1-Y) * np.log(1-A))
        L = np.sum(L_mat, axis = 1) / m
        return L
    def Derivative(self, A, Y):
        assert(A.shape[0] == 1)
        assert(Y.shape[0] == 1)
        assert(A.shape[1] == Y.shape[1])
        dA = ((1-Y)/(1-A)) - (Y/A)        
        return dA
