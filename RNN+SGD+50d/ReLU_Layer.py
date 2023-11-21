# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 21:35:39 2023

@author: fist5
"""

class ReLU_Layer():
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