# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 19:13:51 2023

@author: fist5
"""
import numpy as np

class kernel():
    def __init__(self, input_width, input_height, kernel_size, stride, learning_rate):
        self.input_width = input_width
        self.input_height = input_height
        
        self.kernel_size = kernel_size
        
        self.padding = 0
        self. stride = stride
        
        self.learning_rate = learning_rate
        
        self.output_width = int((self.input_width - self.kernel_size + 2 * self.padding)/self.stride + 1)
        self.output_height = int((self.input_height - self.kernel_size + 2 * self.padding)/self.stride + 1)
        
        self.kernel = np.random.randn(self.kernel_size, self.kernel_size)*0.1
        self.kernel_gradient = np.zeros((self.kernel_size, self.kernel_size))
    
    def inputImg2col(self, Inputimg):
        return Inputimg.reshape(self.input_width * self.input_height, 1)
    
    def col2outputImg(self, col):
        return col.reshape(self.output_height, self.output_width)
    
    def kernel_sparsing(self):
        sparsedKernel = np.zeros((self.output_height*self.output_width, self.input_height*self.input_width))
        pivot = self.kernel.copy()
        for i in range(self.output_width - 1):
            pivot = np.insert(pivot, self.kernel_size + i, 0, axis=1)
        pivot = pivot.flatten()
        for i in range(self.output_width - 1):
            pivot = np.delete(pivot, -1, axis=0)
        for row_num in range(self.output_height*self.output_width):
            for col_num in range(len(pivot)):
                sparsedKernel[row_num][row_num*self.stride + col_num] = pivot[col_num]
        return sparsedKernel
    
    def C_sparsing(self, A, input_height, input_width):
        kernel_height = len(A)
        kernel_width = len(A[0])
        output_width = int((input_width - kernel_width)/self.stride + 1)
        output_height = int((input_height - kernel_height)/self.stride + 1)
        sparsedA = np.zeros((output_height*output_width, input_height*input_width))
        pivot = A.copy()
        for i in range(output_width - 1):
            pivot = np.insert(pivot, kernel_width + i, 0, axis=1)
        pivot = pivot.flatten()
        for i in range(output_width - 1):
            pivot = np.delete(pivot, -1, axis=0)
        for row_num in range(output_height*output_width):
            for col_num in range(len(pivot)):
                sparsedA[row_num][row_num*self.stride + col_num] = pivot[col_num]
        return sparsedA
    
    def convolution2d(self, input):
        col = self.inputImg2col(input)
        sparsedKernel = self.kernel_sparsing()
        col = sparsedKernel @ col 
        return self.col2outputImg(col)
    
    def outputGrad2col(self, output_gradient):
        return output_gradient.reshape(self.output_width * self.output_height, 1)
    
    def col2inputGrad(self, col):
        return col.reshape(self.input_height, self.input_width)
    
    def update(self):
        self.kernel = self.kernel - self.learning_rate * self.kernel_gradient
        self.kernel_gradient = np.zeros((self.kernel_size, self.kernel_size))
        
    def backConvolution2d(self, output_gradient):
        col = self.outputGrad2col(output_gradient)
        sparsedKernel = self.kernel_sparsing()
        col = np.transpose(sparsedKernel) @ col 
        return self.col2inputGrad(col)