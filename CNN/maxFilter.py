# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 00:28:22 2023

@author: fist5
"""
import numpy as np

class maxFilter():
    def __init__(self, input_width, input_height, input_depth, filter_size):
        self.input_width = input_width
        self.input_height = input_height
        self.input_depth = input_depth
        
        self.filter_size = filter_size
        
        self.padding = 0
        self. stride = filter_size
        
        self.output_width = int((self.input_width - self.filter_size + 2 * self.padding)/self.stride + 1)
        self.output_height = int((self.input_height - self.filter_size + 2 * self.padding)/self.stride + 1)
        
        self.extractedIndexMap = np.zeros((self.output_height, self.output_width))
        
    def maxFiltering(self, input):
        input = np.array(input)
        output = np.zeros((self.output_height, self.output_width))
        for row_num in range(self.output_height):
            for col_num in range(self.output_width):
                maxVal = np.max(input[row_num*self.filter_size:row_num*self.filter_size+self.filter_size, col_num*self.filter_size:col_num*self.filter_size+self.filter_size].copy())
                indexVal = np.argmax(input[row_num*self.filter_size:row_num*self.filter_size+self.filter_size, col_num*self.filter_size:col_num*self.filter_size+self.filter_size].copy())
                output[row_num, col_num] = maxVal
                self.extractedIndexMap[row_num, col_num] = (row_num*self.filter_size + int(indexVal/self.filter_size)) * self.input_width + col_num*self.filter_size + indexVal % self.filter_size
        return output
    
    def backFiltering(self, input, output_gradient):
        input_gradient = np.zeros((self.input_height, self.input_width))
        output_gradient = np.array(output_gradient)
        for row_num in range(self.output_height):
            for col_num in range(self.output_width):
                input_gradient[int(self.extractedIndexMap[row_num, col_num]/self.input_width), int(self.extractedIndexMap[row_num, col_num]%self.input_width)] = output_gradient[row_num, col_num]
        return input_gradient