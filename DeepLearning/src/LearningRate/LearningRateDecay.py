# -*- coding: UTF-8 -*-

class DecayClass(object):
    def __init__(self, alpha_0, decayRate):
        self.alpha_0 = alpha_0
        self.decayRate = decayRate
    def calc(self, epochNum):
        return self.alpha_0 / (self.decayRate*epochNum + 1 )


