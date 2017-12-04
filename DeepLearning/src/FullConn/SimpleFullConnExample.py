# -*- coding: UTF-8 -*-
import numpy as np
from Activator.ActivatorsClass import *

y= np.array([[0],
             [1],
             [1],
             [0],
             [0]])

x = np.array([[0,0,1],
              [1,0,1],
              [0,1,1],
              [1,1,1],
              [0,0,1]
              ])

np.random.seed(1)
w0 = np.random.uniform(-1, 1, (3, 4))
w1 = np.random.uniform(-1, 1, (4, 1))

activate = SigmoidActivator()
for i in range(5000):
    l0 = x
    l1 = activate.forward(np.dot(l0,w0))
    #print('l1:',l1)
    #print('l1.T:',l1.T)
    l2 = activate.forward(np.dot(l1,w1))

    l2_err= y-l2
    if (i%1000 == 0):
        print('Error'+ str(np.mean(np.abs(l2_err))))
        
    l2_del = l2_err * activate.backward(l2)
    l1_err = l2_del.dot(w1.T)
    l1_del= l1_err * activate.backward(l1)

    w1 += l1.T.dot(l2_del)
    w0 += l0.T.dot(l1_del)
