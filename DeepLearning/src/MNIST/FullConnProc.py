# -*- coding: UTF-8 -*-
from MNIST.GetDataSetClass import *
from Activator.ActivatorsClass import *
from FullConn.NetworkClass import FullConnNetwork

file1 = 'E:/eclipse/eclipse-workspace/MNIST/train-images-idx3-ubyte'
file2 = 'E:/eclipse/eclipse-workspace/MNIST/train-labels-idx1-ubyte'
file3 = 'E:/eclipse/eclipse-workspace/MNIST/t10k-images-idx3-ubyte'
file4 = 'E:/eclipse/eclipse-workspace/MNIST/t10k-labels-idx1-ubyte'

s = 784
m_test = 1000
epoch = 500
learn_rate = 0.1
minibatch_size = 512
test_minibatch_size = 10000
activator = SigmoidActivator()
n = [40,30,20,10]
TrainSet = MNIST_getDataSet(file1, file2)
X, Y, minibatch_num = TrainSet.minibatch(minibatch_size)
print('minibatch_num',minibatch_num)

network = FullConnNetwork(s, activator, learn_rate, n)

print('Training start')
for i in range(epoch):
    if i % (epoch // 10) == 0:
        progress = i / epoch *100
        print('Training progress: ', progress,'%')
    if i == (epoch-1):
        print('Training progress: 100%')
    for j in range(minibatch_num):
        network.Train(X[j], Y[j])

print('Training end\nTesting start')
TrainnSet_test, TrainnSet_Lable, test_minibatch_num =TrainSet.minibatch(test_minibatch_size)
TrainTest_result = network.ForwardPropagation(TrainnSet_test[3])
accuracy = evaluate(TrainTest_result, TrainnSet_Lable[3])
print('TrainSet predict accuracy', accuracy)

TestSet = MNIST_getDataSet(file3, file4)
X_test, Y_test, test_minibatch_num = TestSet.minibatch(test_minibatch_size)
TestSet_result = network.ForwardPropagation(X_test[0])
accuracy = evaluate(TestSet_result, Y_test[0])
print('TestSet predict accuracy', accuracy)




