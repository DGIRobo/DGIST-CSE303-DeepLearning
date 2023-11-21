# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 22:34:00 2023

@author: fist5
"""
class ReLU():
    def forward(self, input):
        output = input.copy()
        deterMat = (input > 0)
        output = output*deterMat
        return output
    
    def backward(self, input, output_gradient):
        input_gradient = output_gradient.copy()
        deterMat = (input > 0)
        input_gradient = input_gradient*deterMat
        return input_gradient