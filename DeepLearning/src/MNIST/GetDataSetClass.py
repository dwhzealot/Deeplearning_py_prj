# -*- coding: UTF-8 -*-
import numpy as np  
import struct  
from NormalizeInput.NormorlizeInputClass import *
from MyTensorFlow.TF_NormalizeClass import *

MNIST_TrainDataSet = 'E:/eclipse/eclipse-workspace/MNIST/train-images-idx3-ubyte'
MNIST_TrainLabelSet = 'E:/eclipse/eclipse-workspace/MNIST/train-labels-idx1-ubyte'
MNIST_TestDataSet = 'E:/eclipse/eclipse-workspace/MNIST/t10k-images-idx3-ubyte'
MNIST_TestLabelSet = 'E:/eclipse/eclipse-workspace/MNIST/t10k-labels-idx1-ubyte'

class MNIST_getDataSet(object):  
    def __init__(self, dataFileName, labelFileName):
        # 读取二进制文件 
        dataBinfile = open(dataFileName, 'rb')  
        dataBuffers = dataBinfile.read()  
        dataHead = struct.unpack_from('>IIII', dataBuffers, 0) # 取前4个整数，返回一个元组  
        dataImgNum = dataHead[1]
        
        labelBinfile = open(labelFileName, 'rb') # 读二进制文件  
        labelBuffers = labelBinfile.read()  
        labelHead = struct.unpack_from('>II', labelBuffers, 0) # 取label文件前2个整形数  
        labelNum = labelHead[1]

        assert(dataImgNum == labelNum)
        self.totalSampleNum = dataImgNum

        dataOffset = struct.calcsize('>IIII')  # 定位到data开始的位置  
        width = dataHead[2]  
        height = dataHead[3]  
      
        bits = dataImgNum * width * height  # data一共有60000*28*28个像素值  
        bitsString = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'  
      
        dataImgs = struct.unpack_from(bitsString, dataBuffers, dataOffset) # 取data数据，返回一个元组  
        dataBinfile.close()  
        dataImgs = np.reshape(dataImgs,[dataImgNum, width * height]) # reshape为[60000,784]型数组  
        self.dataSet = dataImgs.T
        
        labelOffset = struct.calcsize('>II')  # 定位到label数据开始的位置  
        numString = '>' + str(labelNum) + "B" # fmt格式：'>60000B'  
        labels = struct.unpack_from(numString, labelBuffers, labelOffset) # 取label数据  
      
        labelBinfile.close()  
        label_vec = []
        for i in range(labelNum):
            for j in range(10):
                if j == labels[i]:
                    label_vec.append(0.9)
                else:
                    label_vec.append(0.1)

        labels = np.reshape(label_vec,[labelNum, 10]) # 转型为列表(一维数组)  
        self.labelSet = labels.T

        self.NmlzMean = 0
        self.NmlzVariance = 0
        
    def minibatch(self, minibatch_size, Nmlz = False, Train = True):
        minibatch_num = self.totalSampleNum // minibatch_size
        rest_num = self.totalSampleNum % minibatch_size
        
        if Nmlz == True :
            RawData = np.copy(self.dataSet)
            if Train == True :
                self.NmlzMean,  self.NmlzVariance = TF_GetMeanVariance(RawData)
            
            #assert((self.NmlzMean != 0) | (self.NmlzVariance != 0))
            self.dataSet = TF_NormalizeData(RawData, self.NmlzMean, self.NmlzVariance)
       
        X = []
        offset = 0
        for i in range(minibatch_num):
            minibatch = self.dataSet[:, offset:(offset+ minibatch_size)]
            minibatch = np.reshape(minibatch, [784, minibatch_size])
            X.append(minibatch)
            offset += minibatch_size

        Y = []
        offset = 0
        for j in range(minibatch_num):
            minibatch_Y = self.labelSet[:, offset:(offset+ minibatch_size)]
            minibatch_Y = np.reshape(minibatch_Y, [10, minibatch_size])
            Y.append(minibatch_Y)
            offset += minibatch_size
        
        if (rest_num != 0):
            minibatch_rest = self.dataSet[:, (self.totalSampleNum - rest_num):self.totalSampleNum]
            X.append(minibatch_rest)
        
            minibatch_Y_rest = self.labelSet[:, (self.totalSampleNum - rest_num):self.totalSampleNum]
            Y.append(minibatch_Y_rest)
        
            minibatch_num += 1
        
        #print("minibatch_num: %d; rest:%d" %(minibatch_num, rest_num))
        return X, Y, minibatch_num

    def random_block(self, sample_num):
        block_num = self.totalSampleNum // sample_num
        block_id = np.random.randint(block_num)
        dateBlockSet = self.dataSet[:, block_id:(block_id + sample_num)]
        #print('random_block',dateBlockSet.shape)
        dateBlockSet = np.reshape(dateBlockSet, [784, sample_num])
        
        labelBlockSet = self.labelSet[:, block_id:(block_id + sample_num)]
        #print('random_block',labelBlockSet.shape)
        labelBlockSet = np.reshape(labelBlockSet, [10, sample_num])
        return dateBlockSet, labelBlockSet
    
    

def loadImageSet(filename, img_num):  
    # 读取二进制文件 
    binfile = open(filename, 'rb')  
    buffers = binfile.read()  
  
    head = struct.unpack_from('>IIII', buffers, 0) # 取前4个整数，返回一个元组  
  
    offset = struct.calcsize('>IIII')  # 定位到data开始的位置  
    imgNum = img_num  #head[1]
    width = head[2]  
    height = head[3]  
  
    bits = imgNum * width * height  # data一共有60000*28*28个像素值  
    bitsString = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'  
  
    imgs = struct.unpack_from(bitsString, buffers, offset) # 取data数据，返回一个元组  
  
    binfile.close()  
    imgs = np.reshape(imgs, [imgNum, width * height]) # reshape为[60000,784]型数组  
  
    return imgs,head
  
  
def loadLabelSet(filename, label_num):  
  
    binfile = open(filename, 'rb') # 读二进制文件  
    buffers = binfile.read()  
  
    head = struct.unpack_from('>II', buffers, 0) # 取label文件前2个整形数  
  
    labelNum = label_num #head[1]  
    offset = struct.calcsize('>II')  # 定位到label数据开始的位置  
  
    numString = '>' + str(labelNum) + "B" # fmt格式：'>60000B'  
    labels = struct.unpack_from(numString, buffers, offset) # 取label数据  
  
    binfile.close()  
    label_vec = []
    for i in range(labelNum):
        for j in range(10):
            if j == labels[i]:
                label_vec.append(0.9)
            else:
                label_vec.append(0.1)

    labels = np.reshape(label_vec, [labelNum,10]) # 转型为列表(一维数组)  
  
    return labels,head

# 随机度去出个数为sample_num的数据块
def mnist_load_random_block(data_filename, label_filename, sample_num):

    datafile = open(data_filename, 'rb')  
    buffers = datafile.read()  
  
    head = struct.unpack_from('>IIII', buffers, 0)  
  
    offset = struct.calcsize('>IIII')  
    imgNum = head[1]
    width = head[2]  
    height = head[3]  
  
    bits = sample_num * width * height    
    bitsString = '>' + str(bits) + 'B'  
  
    block_num = imgNum // sample_num
    block_id = np.random.randint(block_num)
    imgs = struct.unpack_from(bitsString, buffers, offset + (block_id * width * height))
    
    datafile.close()  
    imgs = np.reshape(imgs, [sample_num, width * height]) 
    imgs = imgs.T
  #########################################################################################
    labelfile = open(label_filename, 'rb') 
    buffers = labelfile.read()  
  
    head = struct.unpack_from('>II', buffers, 0)   
  
    labelNum = head[1]  
    offset = struct.calcsize('>II')    
  
    numString = '>' + str(sample_num) + "B"  
    
    labels = struct.unpack_from(numString, buffers, offset + block_id)  
  
    labelfile.close()  

    label_vec = []
    for i in range(sample_num):
        for j in range(10):
            if j == labels[i]:
                label_vec.append(0.9)
            else:
                label_vec.append(0.1)

    label_vec = np.reshape(label_vec, [sample_num,10])  
    labels = label_vec.T
    return imgs, labels
  
def get_result(vec):
    max_value_index = 0
    max_value = 0
    vec_size = vec.shape
    m = vec_size[0]
    result = np.zeros(m)
    for i in range(m):
        max_value = 0
        for j in range(10):
            if vec[i][j] > max_value:
                max_value = vec[i][j]
                result[i] = j
    return result
    

def evaluate(test_result_mat, test_labels_mat):
    test_result = test_result_mat.T
    test_labels = test_labels_mat.T
    correct = 0
    total = len(test_result)
    label = get_result(test_labels)
    predict = get_result(test_result)
    for i in range(total):
        if label[i] == predict[i]:
            correct += 1
    return float(correct) / float(total)



