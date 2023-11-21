# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:15:02 2023

@author: fist5
"""

import numpy as np
import RNN_Node

class RNN_Layer():
    def __init__(self, input_size, output_size, Nodes_at_each_layer, learningRate):
        self.input_size = input_size
        self.output_size = output_size
        self.Nodes_at_each_layer = Nodes_at_each_layer
        self.learningRate = learningRate
        self.RNN_Nodes = []
        for i in range(self.Nodes_at_each_layer):
            self.RNN_Nodes.append(RNN_Node.RNN_Node(self.input_size, self.output_size, self.learningRate))
    
    def forward(self, input):
        output = []
        for i in range(self.Nodes_at_each_layer):
            if i == 0:
                output.append(self.RNN_Nodes[i].forward(input[i], np.zeros((self.output_size, 1))))
            else:
                output.append(self.RNN_Nodes[i].forward(input[i], output[i-1]))
        return output
    
    def backward(self, output_gradient):
        input_gradient = [0]*self.Nodes_at_each_layer
        for i in range(self.Nodes_at_each_layer):
            if i == 0:
                before_h_gradient, current_x_gradient = self.RNN_Nodes[self.Nodes_at_each_layer-i-1].backward(output_gradient[self.Nodes_at_each_layer-i-1] + np.zeros((self.output_size, 1)))
                input_gradient[self.Nodes_at_each_layer-i-1] = current_x_gradient
            else:
                before_h_gradient, current_x_gradient = self.RNN_Nodes[self.Nodes_at_each_layer-i-1].backward(output_gradient[self.Nodes_at_each_layer-i-1] + before_h_gradient)
                input_gradient[self.Nodes_at_each_layer-i-1] = current_x_gradient
        return input_gradient
    
    def update(self):
        for i in range(self.Nodes_at_each_layer):
            self.RNN_Nodes[i].update()