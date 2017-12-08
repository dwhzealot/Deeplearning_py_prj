# -*- coding: UTF-8 -*-
'''
Created on 2017年12月5日

@author: DongWenhao
'''
import numpy as np
def Sigmoid_W_init_coe(n):
    '''
    n : W的列数，即W的特征个数
    '''
    a = 1/n
    return np.sqrt(a)

def Tanh_W_init_coe(n):
    '''
    n : W的列数，即W的特征个数
    '''
    a = 1/n
    return np.sqrt(a)

def ReLU_W_init_coe(n):
    '''
    n : W的列数，即W的特征个数
    '''
    a = 2/n
    return np.sqrt(a)