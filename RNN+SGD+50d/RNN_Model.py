# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:09:44 2023

@author: fist5
"""

import numpy as np
import Embedding_Layer
import RNN_Layer
import Dropout_Layer
import Dense_Layer
import ReLU_Layer

class RNN():
    def __init__(self, Nodes_at_each_layer, learningRate, dropout_rate):
        self.Embedding_Layer = Embedding_Layer.Embedding_Layer(10, 50)
        self.RNN_Layer1 = RNN_Layer.RNN_Layer(50, 128, Nodes_at_each_layer, learningRate)
        self.Dropout_Layer1 = Dropout_Layer.Dropout_Layer(128, 128, Nodes_at_each_layer, dropout_rate)
        self.RNN_Layer2 = RNN_Layer.RNN_Layer(128, 128, Nodes_at_each_layer, learningRate)
        self.Dropout_Layer2 = Dropout_Layer.Dropout_Layer(128, 128, Nodes_at_each_layer, dropout_rate)
        self.Dense_Layer = Dense_Layer.Dense_Layer(128, 5, learningRate)
        self.ReLU_Layer = ReLU_Layer.ReLU_Layer()
        self.forwardHistory = [0, 0, 0, 0, 0, 0, 0]
        self.backwardHistory = [0, 0, 0, 0, 0, 0, 0]
    
    def forward(self, input_sentense):
        self.forwardHistory[0] = input_sentense
        self.forwardHistory[1] = self.Embedding_Layer.forward(self.forwardHistory[0])
        self.forwardHistory[2] = self.RNN_Layer1.forward(self.forwardHistory[1])
        self.forwardHistory[3] = self.Dropout_Layer1.forward(self.forwardHistory[2])
        self.forwardHistory[4] = self.RNN_Layer2.forward(self.forwardHistory[3])
        self.forwardHistory[5] = self.Dropout_Layer2.forward(self.forwardHistory[4])
        self.forwardHistory[6] = self.Dense_Layer.forward(self.forwardHistory[5][-1])
        output = self.ReLU_Layer.forward(self.forwardHistory[6])
        return output
    
    def backward(self, output_gradient):
        self.backwardHistory[0] = output_gradient
        self.backwardHistory[1] = self.ReLU_Layer.backward(self.forwardHistory[6], self.backwardHistory[0])
        self.backwardHistory[2] = self.Dense_Layer.backward(self.forwardHistory[5][-1], self.backwardHistory[1])
        self.backwardHistory[3] = self.Dropout_Layer2.backward([np.zeros((128, 1))]*9+[self.backwardHistory[2]])
        self.backwardHistory[4] = self.RNN_Layer2.backward(self.backwardHistory[3])
        self.backwardHistory[5] = self.Dropout_Layer1.backward(self.backwardHistory[4])
        self.backwardHistory[6] = self.RNN_Layer1.backward(self.backwardHistory[5])
    
    def deactivateDropout(self):
        self.Dropout_Layer1.deactivate()
        self.Dropout_Layer2.deactivate()
    
    def activateDropout(self):
        self.Dropout_Layer1.activate()
        self.Dropout_Layer2.activate()
    
    def update(self):
        self.RNN_Layer1.update()
        self.RNN_Layer2.update()
        self.Dense_Layer.update()
    
    def SoftMax(self, input):
        maxVal = np.max(input)
        sum = np.sum(np.exp(input-maxVal))
        return np.exp(input-maxVal)/sum
    
    def CrossEntropyLoss(self, label, softMaxOutput):
        return -np.matmul(np.transpose(label), np.log(softMaxOutput + 1e-6))
    
    def dotCrossEntropyLoss(self, label, softMaxOutput):
        return softMaxOutput - label