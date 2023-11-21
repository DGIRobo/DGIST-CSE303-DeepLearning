# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:16:32 2023

@author: fist5
"""

import numpy as np
import random

class Dropout_Node():
    def __init__(self, dropout_rate, input_size):
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        self.pass_value = np.ones((self.input_size, 1))
    
    def forward(self, input):
        return self.pass_value * input
    
    def backward(self, output_gradient):
        return self.pass_value * output_gradient
    
    def activate(self):
        self.pass_value = np.random.choice(range(0,2), (self.input_size, 1), p=[self.dropout_rate, 1-self.dropout_rate])
        
    def deactivate(self):
        self.pass_value = (1-self.dropout_rate)*np.ones((self.input_size, 1))