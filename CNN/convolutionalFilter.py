# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 20:54:24 2023

@author: fist5
"""

import numpy as np
import kernel

class convolutionalFilter():
    def __init__(self, input_width, input_height, input_depth, kernel_size, kernel_depth, stride, learning_rate):
        self.input_width = input_width
        self.input_height = input_height
        self.input_depth = input_depth
        
        self.kernel_size = kernel_size
        
        self.output_width = int((self.input_width - self.kernel_size)/stride + 1)
        self.output_height = int((self.input_height - self.kernel_size)/stride + 1)
        self.output_depth = kernel_depth
        
        self.kernels = []
        for channel in range(self.input_depth):
            self.kernels.append(kernel.kernel(input_width, input_height, kernel_size, stride, learning_rate))
            
    def convolution3d(self, inChannels):
        outChannel = np.zeros((self.output_height, self.output_width))
        for kernelId in range(len(self.kernels)):
            outChannel = np.zeros((self.output_height, self.output_width))
            for inChannel in inChannels:
                outChannel = outChannel + self.kernels[kernelId].convolution2d(inChannel)
        return outChannel
    
    def update(self):
        for kernelId in range(len(self.kernels)):
            self.kernels[kernelId].update()