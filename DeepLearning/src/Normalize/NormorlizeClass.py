# -*- coding: UTF-8 -*-
import numpy as np

class Normalize (object):
    def ProcTrainData(self, TrainData):
        m = TrainData.shape[1]
        n = TrainData.shape[0]
        eps = 1e-05
        miu = np.sum(TrainData,axis=1, keepdims = True) / m
        self.miu = np.reshape(miu, (n,1))
        X_minus_miu = TrainData - self.miu
        X_square = np.square(X_minus_miu)
        sigmaSquare = np.sum(X_square,axis=1, keepdims = True) / m
        self.sigmaSquare = np.reshape(sigmaSquare, (n,1))
        self.trainDataRet = X_minus_miu / np.sqrt(self.sigmaSquare + eps)
        return self.trainDataRet
    '''
    def ProcTestData(self, TestData):
        TestData_minus_miu = TestData - self.miu
        return TestData_minus_miu / self.sigmaSquare
    '''

def NormalizeTestSet(TestSet, miu, sigmaSquare):
    TestData_minus_miu = TestSet - miu
    return TestData_minus_miu / sigmaSquare
   
   
'''
A = np.array([[0,1,2,3,4,5,6,7,8,9],
              [100,101,102,103,104,105,106,107,108,109]]
)
Nml = NormalizeInput(A)
print(Nml.miu, Nml.sigmaSquare)
print(Nml.ProcTrainData())

B = np.array([[6,7,8,9],
              [106,107,108,109]]
)
print(Nml.ProcTestData(B))
'''