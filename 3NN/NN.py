# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 22:29:43 2023

@author: fist5
"""
import linear
import ReLU
import numpy as np

class NN():
    def __init__(self, FLNeurons, SLNeurons, learning_rate):
        self.learning_rate = learning_rate
        self.inputs = [0 ,0, 0, 0, 0]
        self.output_gradients = [0 ,0, 0, 0, 0]
        self.linear1 = linear.LinearLayer(28*28, FLNeurons, self.learning_rate) # Set Input Size 1x28^2
        self.linear2 = linear.LinearLayer(FLNeurons, SLNeurons, self.learning_rate) # Set Hidden 1 Size 1x500
        self.linear3 = linear.LinearLayer(SLNeurons, 10, self.learning_rate) # Set Hidden 2 Size 1x200
        self.ReLU = ReLU.ReLU()
    
    def forward(self, input):
        self.inputs[0] = input
        self.inputs[1] = self.linear1.forward(self.inputs[0])
        self.inputs[2] = self.ReLU.forward(self.inputs[1])
        self.inputs[3] = self.linear2.forward(self.inputs[2])
        self.inputs[4] = self.ReLU.forward(self.inputs[3])
        output = self.linear3.forward(self.inputs[4])
        return output
    
    def backward(self, output_gradient):
        self.output_gradients[0] = output_gradient
        self.output_gradients[1] = self.linear3.backward(self.inputs[4], self.output_gradients[0])
        self.output_gradients[2] = self.ReLU.backward(self.inputs[3], self.output_gradients[1])
        self.output_gradients[3] = self.linear2.backward(self.inputs[2], self.output_gradients[2])
        self.output_gradients[4] = self.ReLU.backward(self.inputs[1], self.output_gradients[3])
        self.linear1.backward(self.inputs[0], self.output_gradients[4])
    
    def update(self):
        self.linear1.update()
        self.linear2.update()
        self.linear3.update()
        
    def SoftMax(self, input):
        maxVal = np.max(input)
        sum = np.sum(np.exp(input-maxVal))
        return np.exp(input-maxVal)/sum
    
    def CrossEntropyLoss(self, label, softMaxOutput):
        return -np.matmul(np.transpose(label), np.log(softMaxOutput))
    
    def dotCrossEntropyLoss(self, label, softMaxOutput):
        return softMaxOutput - label 