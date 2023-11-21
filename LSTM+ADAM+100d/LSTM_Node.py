# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:14:25 2023

@author: fist5
"""

import numpy as np

class LSTM_Node():
    def __init__(self, input_size, output_size, learning_rate, beta1, beta2, epsilon):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.input_size = input_size
        self.output_size = output_size
        self.weight = np.random.uniform(-np.sqrt(6/(self.input_size + self.output_size)), np.sqrt(6/(self.input_size + self.output_size)), (4*self.output_size, self.input_size + self.output_size))
        self.bias = np.zeros((4*self.output_size, 1))
        #self.weight_gradient = np.zeros((4*self.output_size, self.input_size + self.output_size))
        #self.bias_gradient = np.zeros((4*self.output_size, 1))
        self.weight_mass = np.zeros((4*self.output_size, self.input_size + self.output_size))
        self.weight_velocity = np.zeros((4*self.output_size, self.input_size + self.output_size))
        self.bias_mass = np.zeros((4*self.output_size, 1))
        self.bias_velocity = np.zeros((4*self.output_size, 1))
        self.stack = np.zeros((self.input_size + self.output_size, 1))
        self.i = np.zeros((self.output_size, 1))
        self.f = np.zeros((self.output_size, 1))
        self.o = np.zeros((self.output_size, 1))
        self.g = np.zeros((self.output_size, 1))
        self.current_c = np.zeros((self.output_size, 1))
        self.before_c = np.zeros((self.output_size, 1))
    
    def forward(self, current_x, before_h, before_c):
        self.before_c = before_c
        self.stack = np.vstack((before_h, current_x))
        ifog = self.weight @ self.stack + self.bias
        self.i = self.sigmoid(np.reshape(ifog[0:self.output_size, 0].copy(), (self.output_size, 1)))
        self.f = self.sigmoid(np.reshape(ifog[self.output_size:2*self.output_size, 0].copy(), (self.output_size, 1)))
        self.o = self.sigmoid(np.reshape(ifog[2*self.output_size:3*self.output_size, 0].copy(), (self.output_size, 1)))
        self.g = np.tanh(np.reshape(ifog[3*self.output_size:4*self.output_size, 0].copy(), (self.output_size, 1)))
        self.current_c = self.f*self.before_c + self.i*self.g
        current_h = self.o*np.tanh(self.current_c)
        return current_h, self.current_c
    
    def backward(self, current_h_gradient, current_c_gradient):
        current_c_gradient = (1 - (np.tanh(self.current_c)*np.tanh(self.current_c)))*self.o*current_h_gradient + current_c_gradient
        before_c_gradient = self.f * current_c_gradient
        f_gradient = (1-self.f) * self.f * (self.before_c * current_c_gradient)
        i_gradient = (1-self.i) * self.i * (self.g * current_c_gradient)
        g_gradient = (1-self.g*self.g) * (self.i * current_c_gradient)
        o_gradient = (1-self.o) * self.o * (np.tanh(self.current_c)*current_h_gradient)
        gradient_stack = np.vstack((i_gradient, f_gradient, o_gradient, g_gradient))
        #self.weight_gradient = self.weight_gradient + (gradient_stack @ np.transpose(self.stack))
        #self.bias_gradient = self.bias_gradient + gradient_stack
        weight_gradient = (gradient_stack @ np.transpose(self.stack))
        bias_gradient = gradient_stack
        self.weight_mass = self.beta1*self.weight_mass + (1-self.beta1)*weight_gradient
        self.weight_velocity = self.beta2*self.weight_velocity + (1-self.beta2)*weight_gradient*weight_gradient
        self.bias_mass = self.beta1*self.bias_mass + (1-self.beta1)*bias_gradient
        self.bias_velocity = self.beta2*self.bias_velocity + (1-self.beta2)*bias_gradient*bias_gradient
        input_gradient = (np.transpose(self.weight) @ gradient_stack)
        before_h_gradient = np.reshape(input_gradient[0:self.output_size, 0].copy(), (self.output_size, 1))
        current_x_gradient = np.reshape(input_gradient[self.output_size:self.input_size + self.output_size, 0].copy(), (self.input_size, 1))
        return before_h_gradient, current_x_gradient, before_c_gradient
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
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
        self.weight_mass = np.zeros((4*self.output_size, self.input_size + self.output_size))
        self.weight_velocity = np.zeros((4*self.output_size, self.input_size + self.output_size))
        self.bias_mass = np.zeros((4*self.output_size, 1))
        self.bias_velocity = np.zeros((4*self.output_size, 1))
        #self.weight_gradient = np.zeros((4*self.output_size, self.input_size + self.output_size))
        #self.bias_gradient = np.zeros((4*self.output_size, 1))