# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:16:41 2023

@author: fist5
"""

import numpy as np
import Dropout_Node

class Dropout_Layer():
    def __init__(self, input_size, output_size, Nodes_at_each_layer, dropout_rate):
        self.input_size = input_size
        self.output_size = output_size
        self.Nodes_at_each_layer = Nodes_at_each_layer
        self.dropout_rate = dropout_rate
        self.Dropout_Nodes = []
        for i in range(self.Nodes_at_each_layer):
            self.Dropout_Nodes.append(Dropout_Node.Dropout_Node(self.dropout_rate, self.input_size))
    
    def forward(self, input):
        output = []
        for i in range(self.Nodes_at_each_layer):
            output.append(self.Dropout_Nodes[i].forward(input[i]))
        return output
    
    def backward(self, output_gradient):
        input_grandient = []
        for i in range(self.Nodes_at_each_layer):
            input_grandient.append(self.Dropout_Nodes[i].backward(output_gradient[i]))
        return input_grandient
    
    def update(self):
        for i in range(self.Nodes_at_each_layer):
            self.Dropout_Nodes[i].update()
            
    def deactivate(self):
        for i in range(self.Nodes_at_each_layer):
            self.Dropout_Nodes[i].deactivate()
            
    def activate(self):
        for i in range(self.Nodes_at_each_layer):
            self.Dropout_Nodes[i].activate()