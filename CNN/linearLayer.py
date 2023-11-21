# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 22:33:25 2023

@author: fist5
"""
import numpy as np

class LinearLayer():
    def __init__(self, inputsize, outputsize, learning_rate):
        self.learning_rate = learning_rate
        self.inputsize = inputsize
        self.outputsize = outputsize
        self.weight = np.random.uniform(-np.sqrt(6/self.inputsize), np.sqrt(6/self.inputsize), (self.outputsize, self.inputsize))
        self.bias = np.zeros((self.outputsize, 1))
        self.weight_gradient = np.zeros((self.outputsize, self.inputsize))*0.1
        self.bias_gradient = np.zeros((self.outputsize, 1))*0.1
    
    def forward(self, input):
        output = (self.weight @ input) + self.bias
        return output
    
    def backward(self, input, output_gradient):
        self.weight_gradient = self.weight_gradient + (output_gradient @ np.transpose(input))
        self.bias_gradient = self.bias_gradient + output_gradient
        input_gradient = (np.transpose(self.weight) @ output_gradient)
        return input_gradient
    
    def update(self):
        self.weight = self.weight - self.learning_rate*self.weight_gradient
        self.bias = self.bias - self.learning_rate*self.bias_gradient
        self.weight_gradient = np.zeros((self.outputsize, self.inputsize))
        self.bias_gradient = np.zeros((self.outputsize, 1))