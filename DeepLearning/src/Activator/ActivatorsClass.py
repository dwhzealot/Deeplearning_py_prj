# -*- coding: UTF-8 -*-
import numpy as np

class SigmoidActivator(object):
    def forward(self, Z):
        return 1.0 / (1.0 + np.exp(-Z))
    def backward(self, Z):
        A = 1.0 / (1.0 + np.exp(-Z))
        return A * (1 - A)

class TanhActivator(object):
    def forward(self, Z):
        return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    def backward(self, Z):
        A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
        return (1 -np.square(A))
    
class ReLUActivator(object):
    def forward(self, Z):
        return np.maximum(Z, 0)
    
    def backward(self, Z):
        A = np.maximum(0,Z)
        A[A > 0] = 1
        return A
