# -*- coding: UTF-8 -*-
import numpy as np
from CostF import *

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
        m     :样本个数
        s     :单样本元素个数
        l_max : 总层数
        W_ini_coe: W初始化系数
        activator: 激活函数
        learn_rate: W, b更新时的学习率 
        n: 记录每层的神经元个数的数组
        '''
        self.m = m
        self.s = s
        self.l_max = l_max #从1开始计算
        self.layer = []
        is_outputlayer = False
        column = 0       
        assert(len(n) == l_max)
        self.n = n
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
    
    def GradientCheck (self):
        '''
        #在梯度检查中，如果每次反向传播不更新W， 计算出的delta会比较小。此函数还是保留了剃度检查中的W的更细                                 
        '''
        CostF = CrossEntropy()
        assert(self.n[self.l_max - 1] == 1)
        X = [1,2,3,4,5]
        X = np.reshape(X, (1,5))
        Y = [0.1,0.2,0.3,0.4,0.5]
        Y = np.reshape(Y, (1,5))
        delta_max = 0
        epsilon = 1e-07
        for l in range(self.l_max):
            for i in range(self.layer[l].W.shape[0]):
                for j in range(self.layer[l].W.shape[1]):
                    W_backup = np.copy(self.layer[l].W)
                    W_add_epsilon = np.copy(self.layer[l].W)
                    W_minus_epsilon = np.copy(self.layer[l].W)
                    W_add_epsilon[i,j] += epsilon
                    W_minus_epsilon[i,j] -= epsilon
                    
                    self.layer[l].W = W_add_epsilon
                    A_W_add_epsilon = self.ForwardPropagation(X)
                    L_W_add_epsilon = CostF.Calculate(A_W_add_epsilon, Y)

                    self.layer[l].W = W_minus_epsilon
                    A_W_minus_epsilon = self.ForwardPropagation(X)
                    L_W_minus_epsilon = CostF.Calculate(A_W_minus_epsilon, Y)                    
                    
                    dW_app = (L_W_add_epsilon - L_W_minus_epsilon) / (2*epsilon)
                    
                    self.layer[l].W = W_backup
                    self.Train(X, Y)
                    dW_calc = self.layer[l].dW[i,j]
                    
                    delta = dW_calc - dW_app
                    delta_abs = np.abs(delta)
                    if delta_abs > delta_max :
                        delta_max = delta_abs
                        delta_max_l = l
                        delta_max_i = i
                        delta_max_j = j
                    if delta > 1e-05 :
                        print("GradientCheck failed; layer[%d]W[%d,%d], delta:%.e" %(l,i,j,delta))
                        return
        print("GradientCheck finish, delta_max: layer[%d]W[%d,%d] %e" %(l,i,j,delta_max))
                       
