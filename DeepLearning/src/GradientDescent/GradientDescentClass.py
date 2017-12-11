# -*- coding: UTF-8 -*-
import numpy as np
class MomentumGradientDescent(object):
    def __init__(self):
        self.beta = 0.9
        self.v_dW = 0
        self.v_db = 0
    def calc(self, dW, db): 
        self.v_dW = (self.beta*self.v_dW) + ((1-self.beta)*dW)
        self.v_db = (self.beta*self.v_db) + ((1-self.beta)*db)
        return self.v_dW, self.v_db
    
class RMSprop(object):
    def __init__(self):
        self.beta = 0.99
        self.S_dW = 0
        self.S_db = 0
    def calc(self, dW, db): 
        epsilon = 1e-08
        self.S_dW = (self.beta*self.S_dW) + ((1-self.beta)*(np.square(dW)))
        self.S_db = (self.beta*self.S_db) + ((1-self.beta)*(np.square(db)))        
        delta_W = dW / (np.sqrt(self.S_dW) + epsilon)
        delta_b = dW / (np.sqrt(self.S_db) + epsilon)
        return delta_W, delta_b
    
class Adam(object):
    def __init__(self):
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.v_dW = 0
        self.v_db = 0
        self.S_dW = 0
        self.S_db = 0
        self.t = 0
    def calc(self, dW, db):
        epsilon = 1e-08
        self.t += 1
        self.v_dW = (self.beta1*self.v_dW) + ((1-self.beta1)*dW)
        self.v_db = (self.beta1*self.v_db) + ((1-self.beta1)*db)
        self.S_dW = (self.beta2*self.S_dW) + ((1-self.beta2)*(np.square(dW)))
        self.S_db = (self.beta2*self.S_db) + ((1-self.beta2)*(np.square(db)))  
        v_dW_correct = self.v_dW / (1 - np.power(self.beta1, self.t))
        v_db_correct = self.v_db / (1 - np.power(self.beta1, self.t))
        S_dW_correct = self.S_dW / (1 - np.power(self.beta1, self.t))
        S_db_correct = self.S_db / (1 - np.power(self.beta1, self.t))  
        delta_W = v_dW_correct / (np.sqrt(S_dW_correct) + epsilon)
        delta_b = v_db_correct / (np.sqrt(S_db_correct) + epsilon)
        return delta_W, delta_b       
        
        
              
        