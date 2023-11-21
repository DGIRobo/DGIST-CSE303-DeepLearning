# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 22:35:36 2023

@author: fist5
"""
import numpy as np
import convolutionalLayer as cL
import poolingLayer as pL
import linearLayer
import ReLULayer

class CNN():
    def __init__(self, input_height, input_width, input_depth, output_size, FLkernelSize, FLkernelNums, SLkernelSize, SLkernelNums, learning_rate):
        self.conv1 = cL.convolutionalLayer(input_width, input_height, input_depth, FLkernelSize, FLkernelNums, 1, learning_rate)
        self.maxpool1 = pL.poolingLayer(self.conv1.output_width, self.conv1.output_height, self.conv1.output_depth, 2)
        self.conv2 = cL.convolutionalLayer(self.maxpool1.output_width, self.maxpool1.output_height, self.maxpool1.output_depth, SLkernelSize, SLkernelNums, 1, learning_rate)
        self.maxpool2 = pL.poolingLayer(self.conv2.output_width, self.conv2.output_height, self.conv2.output_depth, 2)
        self.linear = linearLayer.LinearLayer(self.maxpool2.output_width * self.maxpool2.output_height * self.maxpool2.output_depth, output_size, learning_rate)
        self.ReLULayer = ReLULayer.ReLULayer()
        self.forwardHistory = [0, 0, 0, 0, 0, 0, 0]
        self.backwardHistory = [0, 0, 0, 0, 0, 0, 0]
    
    def forward(self, inChannels):
        self.forwardHistory[0] = inChannels
        self.forwardHistory[1] = self.conv1.forward(self.forwardHistory[0])
        self.forwardHistory[2] = self.ReLULayer.forward(self.forwardHistory[1])
        self.forwardHistory[3] = self.maxpool1.forward(self.forwardHistory[2])
        self.forwardHistory[4] = self.conv2.forward(self.forwardHistory[3])
        self.forwardHistory[5] = self.ReLULayer.forward(self.forwardHistory[4])
        self.forwardHistory[6] = self.maxpool2.forward(self.forwardHistory[5])
        self.forwardHistory[6] = np.reshape(np.array(self.forwardHistory[6]), (self.maxpool2.output_width * self.maxpool2.output_height * self.maxpool2.output_depth, 1))
        output = self.linear.forward(self.forwardHistory[6])
        return output
    
    def backward(self, output_gradients):
        self.backwardHistory[0] = output_gradients
        self.backwardHistory[1] = np.reshape(self.linear.backward(self.forwardHistory[6], self.backwardHistory[0]), (self.maxpool2.output_depth, self.maxpool2.output_height, self.maxpool2.output_width)).tolist()
        for img in range(len(self.backwardHistory[1])):
            self.backwardHistory[1][img] = np.array(self.backwardHistory[1][img])
        self.backwardHistory[2] = self.maxpool2.backward(self.forwardHistory[5], self.backwardHistory[1])
        self.backwardHistory[3] = self.ReLULayer.backward(self.forwardHistory[4], self.backwardHistory[2])
        self.backwardHistory[4] = self.conv2.backward(self.forwardHistory[3], self.backwardHistory[3])
        self.backwardHistory[5] = self.maxpool1.backward(self.forwardHistory[2], self.backwardHistory[4])
        self.backwardHistory[6] = self.ReLULayer.backward(self.forwardHistory[1], self.backwardHistory[5])
        self.conv1.backward(self.forwardHistory[0], self.backwardHistory[6])
    
    def update(self):
        self.conv1.update()
        self.conv2.update()
        self.linear.update()
        
    def SoftMax(self, input):
        maxVal = np.max(input)
        sum = np.sum(np.exp(input-maxVal))
        return np.exp(input-maxVal)/sum
    
    def CrossEntropyLoss(self, label, softMaxOutput):
        return -np.matmul(np.transpose(label), np.log(softMaxOutput + 1e-6))
    
    def dotCrossEntropyLoss(self, label, softMaxOutput):
        return softMaxOutput - label 