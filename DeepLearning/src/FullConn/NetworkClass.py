# -*- coding: UTF-8 -*-
import numpy as np

class FullConnLayer(object):
    def __init__(self, m, W_ini_coe ,activator, learn_rate, is_OutputLayer, n,column):
        '''
        #构造函数
        m: 样本个数
        n:本层神经元个数，即行数
        colum: W的列数 
        W_ini_coe: W初始化系数
        activator: 激活函数
        learn_rate: W, b更新时的学习率 
        OutputLayer_flag: 是否是输出层
        '''
        self.m = m
        self.n = n
        self.column = column
        self.activator = activator
        #np.random.seed(n)
        self.W = W_ini_coe * (np.random.randn(n,column))
        self.W_before_update = np.zeros([n,column])
        self.b = np.zeros((n, 1))
        self.learn_rate = learn_rate
        self.is_OutputLayer = is_OutputLayer
    def forward(self, input_mat):
        '''
        #前向计算
        input_mat: 本层的输入, 即为上一层的输出 A
        '''
        self.input = input_mat
        self.Z = np.dot(self.W, input_mat) + self.b
        self.A = self.activator.forward(self.Z)
        self.W_before_update = self.W
        return self.A
    
    def backward(self, dZ_next_layer, W_next_layer, Y):
        '''
        #反向计算W和b的梯度
        dZ_next_layer: 下一层反向传播过来的 dZ
        W_next_layer: 下一层的权重W
        Y: 标签值
        '''
        if (self.is_OutputLayer == True):
            self.dZ = self.A - Y
        else:
            self.dZ = (np.dot(W_next_layer.T,dZ_next_layer)) * (self.activator.backward(self.Z))
        
        dW = (np.dot(self.dZ, self.input.T))/self.m
        db = (np.sum(self.dZ, axis=1, keepdims=True))/self.m
        self.dW = dW
        self.W -= self.learn_rate * dW
        self.b -= self.learn_rate * db
        return self.dZ, self.W_before_update

class FullConnNetwork (object):
    def __init__(self, m, s, l_max, W_ini_coe ,activator, learn_rate,n):
        '''
        (self, m, n, W_ini_coe ,activator, learn_rate, column)
        m     :样本个数
        s     :单样本元素个数
        l_max : 总层数
        W_ini_coe: W初始化系数
        activator: 激活函数
        learn_rate: W, b更新时的学习率 
        n: 每层神经元个数
        '''
        self.m = m
        self.s = s
        self.l_max = l_max #从1开始计算
        self.layer = []
        is_outputlayer = False
        column = 0       
        assert(len(n) == l_max)
        
        if l_max == 1:
            layer = FullConnLayer(m, W_ini_coe ,activator, learn_rate, True, n[0], s)    
            self.layer.append(layer)
            return
        for i in range(l_max):
            if i < (l_max-1):
                #隐藏层
                if (i == 0):
                    #第1层
                    column = self.s
                else:
                    column = n[i-1]
                
            else:
                #输出层
                is_outputlayer = True
                column = n[i-1]
            layer = FullConnLayer(m, W_ini_coe ,activator, learn_rate, is_outputlayer, n[i], column)    
            self.layer.append(layer)
    def ForwardPropagation (self, X):
        A = X
        for i in range(self.l_max):
            A = self.layer[i].forward(A)
        return A
    def BackwardPropagation(self, Y):
        W_next_layer = 0
        dZ_next_layer = 0
        for i in range(self.l_max):
            dZ_next_layer, W_next_layer = self.layer[self.l_max - i -1].backward(dZ_next_layer, W_next_layer, Y)
        return
    def Train(self, X, Labels):
        self.ForwardPropagation(X)
        self.BackwardPropagation(Labels)
            
