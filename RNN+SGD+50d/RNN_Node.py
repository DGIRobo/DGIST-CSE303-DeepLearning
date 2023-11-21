# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:14:25 2023

@author: fist5
"""

import numpy as np

class RNN_Node():
    def __init__(self, input_size, output_size, learning_rate):
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.output_size = output_size
        self.weight = np.random.uniform(-np.sqrt(6/(self.input_size + self.output_size)), np.sqrt(6/(self.input_size + self.output_size)), (self.output_size, self.input_size + self.output_size))
        self.bias = np.zeros((self.output_size, 1))
        self.weight_gradient = np.zeros((self.output_size, self.input_size + self.output_size))
        self.bias_gradient = np.zeros((self.output_size, 1))
        self.stack = np.zeros((self.input_size + self.output_size, 1))
        self.current_h = np.zeros((self.output_size, 1))
    
    def forward(self, current_x, before_h):
        self.stack = np.vstack((before_h, current_x))
        self.currnet_h = np.tanh(self.weight @ self.stack + self.bias)
        return self.currnet_h
    
    def backward(self, current_h_gradient):
        current_h_gradient = (1 - self.current_h*self.current_h)*current_h_gradient
        self.weight_gradient = self.weight_gradient + (current_h_gradient @ np.transpose(self.stack))
        self.bias_gradient = self.bias_gradient + current_h_gradient
        input_gradient = (np.transpose(self.weight) @ current_h_gradient)
        before_h_gradient = np.reshape(input_gradient[0:self.output_size, 0].copy(), (self.output_size, 1))
        current_x_gradient = np.reshape(input_gradient[self.output_size:self.input_size + self.output_size, 0].copy(), (self.input_size, 1))
        return before_h_gradient, current_x_gradient
    
    def update(self):
        self.weight = self.weight - self.learning_rate*self.weight_gradient
        self.bias = self.bias - self.learning_rate*self.bias_gradient
        self.weight_gradient = np.zeros((self.output_size, self.input_size + self.output_size))
        self.bias_gradient = np.zeros((self.output_size, 1))