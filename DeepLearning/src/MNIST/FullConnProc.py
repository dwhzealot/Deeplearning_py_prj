# -*- coding: UTF-8 -*-
from MNIST.GetDataSetClass import *
from Activator.ActivatorsClass import *
from FullConn.NetworkClass import FullConnNetwork

file1 = 'E:/eclipse/eclipse-workspace/MNIST/train-images-idx3-ubyte'
file2 = 'E:/eclipse/eclipse-workspace/MNIST/train-labels-idx1-ubyte'
file3 = 'E:/eclipse/eclipse-workspace/MNIST/t10k-images-idx3-ubyte'
file4 = 'E:/eclipse/eclipse-workspace/MNIST/t10k-labels-idx1-ubyte'


s = 784
m = 1000
m_test = 1000
epoch = 3000
learn_rate = 0.1

TrainSet = MNIST_getDataSet(file1, file2)

print('Dnn3 start')
activator = SigmoidActivator()
n = [30,20,10]

network = FullConnNetwork(m, s, activator, learn_rate, n)

print('Training start')
for i in range(epoch):
    X, Y = TrainSet.random_block(m)
    network.Train(X, Y)

print('Training end\nTesting start')
TestSet = MNIST_getDataSet(file3, file4)
X_test,Lable_test = TestSet.random_block(m_test)
Test_result = network.ForwardPropagation(X_test)

print(evaluate(Test_result, Lable_test))
print('testing end')

print('Dnn3 end')
