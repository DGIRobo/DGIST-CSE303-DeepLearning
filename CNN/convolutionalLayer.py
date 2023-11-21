# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 17:01:51 2023

@author: fist5
"""
import numpy as np
import convolutionalFilter

class convolutionalLayer():
    def __init__(self, input_width, input_height, input_depth, kernel_size, kernel_depth, stride, learning_rate):
        self.input_width = input_width
        self.input_height = input_height
        self.input_depth = input_depth
        
        self.kernel_size = kernel_size
        
        self.output_width = int((self.input_width - self.kernel_size)/stride + 1)
        self.output_height = int((self.input_height - self.kernel_size)/stride + 1)
        self.output_depth = kernel_depth
        
        self.filters = []
        for channel in range(self.output_depth):
            self.filters.append(convolutionalFilter.convolutionalFilter(input_width, input_height, input_depth, kernel_size, kernel_depth, stride, learning_rate))
        
        self.biases = []
        for channel in range(self.output_depth):
            self.biases.append(np.zeros((self.output_height, self.output_width)))
        
        self.bias_gradients = []
        for channel in range(self.output_depth):
            self.bias_gradients.append(np.zeros((self.output_height, self.output_width)))
            
        self.learning_rate = learning_rate
    
    def forward(self, inChannels):
        outChannels = []
        for filterId in range(len(self.filters)):
            outChannel = self.filters[filterId].convolution3d(inChannels)
            outChannel = outChannel + self.biases[filterId]
            outChannels.append(outChannel)
        return outChannels
    
    def backward(self, inChannels, outChannel_gradients):
        # bias_gradient update
        for biasId in range(len(self.bias_gradients)):
            self.bias_gradients[biasId] = self.bias_gradients[biasId] + outChannel_gradients[biasId] 
        # kernel_gradient update 
        for outputId in range(len(outChannel_gradients)):
            for inputId in range(len(inChannels)):
                instantaneous_kernel_gradient = self.filters[outputId].kernels[inputId].C_sparsing(outChannel_gradients[outputId], len(inChannels[inputId]), len(inChannels[inputId][0])) @ self.filters[outputId].kernels[inputId].inputImg2col(inChannels[inputId])
                self.filters[outputId].kernels[inputId].kernel_gradient = np.reshape(instantaneous_kernel_gradient, (self.kernel_size, self.kernel_size))
        # inChannel_gradient update
        inChannel_gradients = []
        for inputId in range(len(inChannels)):
            inChannel_gradient = np.zeros((self.input_height, self.input_width))
            for outputId in range(len(outChannel_gradients)):
                inChannel_gradient = inChannel_gradient + self.filters[outputId].kernels[inputId].backConvolution2d(outChannel_gradients[outputId])
            inChannel_gradients.append(inChannel_gradient)
        return inChannel_gradients
        
    def update(self):
        for filterId in range(len(self.filters)):
            self.filters[filterId].update()
        for biasId in range(len(self.bias_gradients)):
            self.biases[biasId] = self.biases[biasId] - self.learning_rate * self.bias_gradients[biasId]
            self.bias_gradients[biasId] = np.zeros((self.output_height, self.output_width))
        
                
    
    
    
    