# -*- coding: UTF-8 -*-
import numpy as np
from CostFunc.CrossEntropyClass import *
from WeightInit.WeightInitClass import *
class FullConnLayer(object):
    def __init__(self, activator, learn_rate, is_OutputLayer, n,column):
        '''
        #构造函数
        n:本层神经元个数，即行数
        colum: W的列数 
        W_ini_coe: W初始化系数
        activator: 激活函数
        learn_rate: W, b更新时的学习率 
        OutputLayer_flag: 是否是输出层
        '''
        self.n = n
        self.column = column
        self.activator = activator
        #np.random.seed(n)
        W_ini_coe = Sigmoid_W_init_coe(column)
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
        self.m = input_mat.shape[1]  #样本个数
        self.Z = np.dot(self.W, input_mat) + self.b
        self.A = self.activator.forward(self.Z)
        self.W_before_update = np.copy(self.W)
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
        self.db = db
        self.W -= self.learn_rate * dW
        self.b -= self.learn_rate * db
        return self.dZ, self.W_before_update

    def backwardForGradientCheck(self, dZ_next_layer, W_next_layer, Y):
        '''
        #用于梯度检测中的反向计算W和b的梯度, 不更新W和b
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
        self.db = db
        return self.dZ, self.W_before_update
    
class FullConnNetwork (object):
    def __init__(self, s, activator, learn_rate,n):
        '''
        s     :单样本元素个数
        l_max : 总层数
        W_ini_coe: W初始化系数
        activator: 激活函数
        learn_rate: W, b更新时的学习率 
        n: 记录每层的神经元个数的数组
        '''
        self.s = s
        self.l_max = len(n) #从1开始计算
        self.layer = []
        is_outputlayer = False
        column = 0       
        self.n = n
        l_max = self.l_max
        if l_max == 1:
            layer = FullConnLayer(activator, learn_rate, True, n[0], s)    
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
            layer = FullConnLayer(activator, learn_rate, is_outputlayer, n[i], column)    
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

    def BackwardPropagationForGradientCheck(self, Y):
        W_next_layer = 0
        dZ_next_layer = 0
        for i in range(self.l_max):
            dZ_next_layer, W_next_layer = self.layer[self.l_max - i -1].backwardForGradientCheck(dZ_next_layer, W_next_layer, Y)
        return
    
    def GradientCheck (self):
        '''
        #在梯度检查中，每次反向传播不要更新W，否则会导致dW的计算出现问题                           
        '''
        CostF = CrossEntropy()
        assert(self.n[self.l_max - 1] == 1)
        X = [1,2,3,4,5]
        X = np.reshape(X, (1,5))
        Y = [0.1,0.2,0.3,0.4,0.5]
        Y = np.reshape(Y, (1,5))

        self.ForwardPropagation(X)
        self.BackwardPropagationForGradientCheck(Y)
        
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
                    dW_calc = self.layer[l].dW[i,j]
                    
                    delta = dW_calc - dW_app
                    delta_abs = np.abs(delta)
                    if delta_abs > delta_max :
                        delta_max = delta_abs
                        delta_max_l = l
                        delta_max_i = i
                        delta_max_j = j
                    if delta_abs > 1e-05 :
                        print("GradientCheck failed; layer[%d]W[%d,%d], delta:%.e" %(l,i,j,delta_abs))
                        return
        print("GradientCheck finish, delta_max: layer[%d]W[%d,%d] %e" %(delta_max_l,delta_max_i,delta_max_j,delta_max))
                       


