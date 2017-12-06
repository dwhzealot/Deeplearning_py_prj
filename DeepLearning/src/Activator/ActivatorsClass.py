# -*- coding: UTF-8 -*-
import numpy as np

class SigmoidActivator(object):
    def forward(self, Z):
        return 1.0 / (1.0 + np.exp(-Z))
    def backward(self, Z):
        return Z * (1 - Z)

class TanhActivator(object):
    def forward(self, Z):
        return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    def backward(self, Z):
        return (1 -np.square(Z))
    
class ReLUActivator(object):
    def forward(self, Z):
        return np.maximum(Z, 0)
    
    def backward(self, Z):
        Z = np.maximum(0,Z)
        Z[Z > 0] = 1
        return Z
