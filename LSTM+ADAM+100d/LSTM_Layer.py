# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:15:02 2023

@author: fist5
"""

import numpy as np
import LSTM_Node

class LSTM_Layer():
    def __init__(self, input_size, output_size, Nodes_at_each_layer, learningRate, beta1, beta2, epsilon):
        self.input_size = input_size
        self.output_size = output_size
        self.Nodes_at_each_layer = Nodes_at_each_layer
        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.LSTM_Nodes = []
        for i in range(self.Nodes_at_each_layer):
            self.LSTM_Nodes.append(LSTM_Node.LSTM_Node(self.input_size, self.output_size, self.learningRate, self.beta1, self.beta2, self.epsilon))
    
    def forward(self, input):
        output = []
        for i in range(self.Nodes_at_each_layer):
            if i == 0:
                currnet_h, current_c = self.LSTM_Nodes[i].forward(input[i], np.zeros((self.output_size, 1)), np.zeros((self.output_size, 1)))
                output.append(currnet_h)
            else:
                currnet_h, current_c = self.LSTM_Nodes[i].forward(input[i], currnet_h, current_c)
                output.append(currnet_h)
        return output
    
    def backward(self, output_gradient):
        input_gradient = [0]*self.Nodes_at_each_layer
        for i in range(self.Nodes_at_each_layer):
            if i == 0:
                before_h_gradient, current_x_gradient, before_c_gradient = self.LSTM_Nodes[self.Nodes_at_each_layer-i-1].backward(output_gradient[self.Nodes_at_each_layer-i-1] + np.zeros((self.output_size, 1)), np.zeros((self.output_size, 1)))
                input_gradient[self.Nodes_at_each_layer-i-1] = current_x_gradient
            else:
                before_h_gradient, current_x_gradient, before_c_gradient = self.LSTM_Nodes[self.Nodes_at_each_layer-i-1].backward(output_gradient[self.Nodes_at_each_layer-i-1] + before_h_gradient, before_c_gradient)
                input_gradient[self.Nodes_at_each_layer-i-1] = current_x_gradient
        return input_gradient
    
    def update(self):
        for i in range(self.Nodes_at_each_layer):
            self.LSTM_Nodes[i].update()