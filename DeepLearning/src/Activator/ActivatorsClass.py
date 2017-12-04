# -*- coding: UTF-8 -*-
import numpy as np

class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))
    def backward(self, output):
        return output * (1 - output)

class TanhActivator(object):
    def forward(self, weighted_input):
        return (np.exp(weighted_input) - np.exp(-weighted_input)) / (np.exp(weighted_input) + np.exp(-weighted_input))
    def backward(self, output):
        return (1 -np.square(output))
    
class ReLUActivator(object):
    def forward(self, weighted_input):
        if weighted_input > 0 :
            return weighted_input
        else :
            return 0
    def backward(self, output):
        if output > 0 :
            return 1
        else :
            return 0
