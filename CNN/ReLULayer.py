# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 22:34:00 2023

@author: fist5
"""
class ReLULayer():
    def forward(self, inChannels):
        outChannels = inChannels.copy()
        for outputId in range(len(outChannels)):
            deterMat = (inChannels[outputId] > 0)
            outChannels[outputId] = outChannels[outputId]*deterMat
        return outChannels
    
    def backward(self, inChannels, output_gradients):
        input_gradients = output_gradients.copy()
        for InputId in range(len(input_gradients)):
            deterMat = (inChannels[InputId] > 0)
            input_gradients[InputId] = input_gradients[InputId]*deterMat
        return input_gradients