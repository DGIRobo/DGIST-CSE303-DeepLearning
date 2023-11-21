# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 21:16:34 2023

@author: fist5
"""

import numpy as np

class Dense_Layer():
    def __init__(self, inputsize, outputsize, learning_rate, beta1, beta2, epsilon):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.inputsize = inputsize
        self.outputsize = outputsize
        self.weight = np.random.uniform(-np.sqrt(6/self.inputsize), np.sqrt(6/self.inputsize), (self.outputsize, self.inputsize))
        self.bias = np.zeros((self.outputsize, 1))
        self.weight_mass = np.zeros((self.outputsize, self.inputsize))
        self.weight_velocity = np.zeros((self.outputsize, self.inputsize))
        self.bias_mass = np.zeros((self.outputsize, 1))
        self.bias_velocity = np.zeros((self.outputsize, 1))
        #self.weight_gradient = np.zeros((self.outputsize, self.inputsize))
        #self.bias_gradient = np.zeros((self.outputsize, 1))
    
    def forward(self, input):
        output = (self.weight @ input) + self.bias
        return output
    
    def backward(self, input, output_gradient):
        #weight_gradient = self.weight_gradient + (output_gradient @ np.transpose(input))
        #bias_gradient = self.bias_gradient + output_gradient
        weight_gradient = (output_gradient @ np.transpose(input))
        bias_gradient = output_gradient
        self.weight_mass = self.beta1*self.weight_mass + (1-self.beta1)*weight_gradient
        self.weight_velocity = self.beta2*self.weight_velocity + (1-self.beta2)*weight_gradient*weight_gradient
        self.bias_mass = self.beta1*self.bias_mass + (1-self.beta1)*bias_gradient
        self.bias_velocity = self.beta2*self.bias_velocity + (1-self.beta2)*bias_gradient*bias_gradient
        input_gradient = (np.transpose(self.weight) @ output_gradient)
        return input_gradient
    
    def update(self):
        # weight update
        weight_m_hat = self.weight_mass / (1 - self.beta1)
        weight_v_hat = self.weight_velocity / (1 - self.beta2)
        self.weight = self.weight - self.learning_rate*(weight_m_hat / (np.sqrt(weight_v_hat) + self.epsilon))
        # bias update
        bias_m_hat = self.bias_mass / (1 - self.beta1)
        bias_v_hat = self.bias_velocity / (1 - self.beta2)
        self.bias = self.bias - self.learning_rate*(bias_m_hat / (np.sqrt(bias_v_hat) + self.epsilon))
        # gradient reset
        #self.weight_gradient = np.zeros((self.outputsize, self.inputsize))
        #self.bias_gradient = np.zeros((self.outputsize, 1))
        self.weight_mass = np.zeros((self.outputsize, self.inputsize))
        self.weight_velocity = np.zeros((self.outputsize, self.inputsize))
        self.bias_mass = np.zeros((self.outputsize, 1))
        self.bias_velocity = np.zeros((self.outputsize, 1))