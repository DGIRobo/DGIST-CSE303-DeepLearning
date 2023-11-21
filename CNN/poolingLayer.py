# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 17:02:54 2023

@author: fist5
"""
import maxFilter

class poolingLayer():
    def __init__(self, input_width, input_height, input_depth, filter_size):
        self.input_width = input_width
        self.input_height = input_height
        self.input_depth = input_depth
        
        self.filter_size = filter_size
        self.stride = filter_size
        
        self.output_width = int((self.input_width - self.filter_size)/self.stride + 1)
        self.output_height = int((self.input_height - self.filter_size)/self.stride + 1)
        self.output_depth = input_depth
        
        self.filters = []
        for filterId in range(self.input_depth):
            self.filters.append(maxFilter.maxFilter(input_width, input_height, input_depth, filter_size))
        
    def forward(self, inChannels):
        outChannels = []
        for numbering in range(len(inChannels)):
            outChannel = self.filters[numbering].maxFiltering(inChannels[numbering])
            outChannels.append(outChannel)
        return outChannels
    
    def backward(self, inChannels, outChannel_gradients):
        inChannel_gradients = []
        for numbering in range(len(outChannel_gradients)):
            inChannel_gradient = self.filters[numbering].backFiltering(inChannels[numbering], outChannel_gradients[numbering])
            inChannel_gradients.append(inChannel_gradient)
        return inChannel_gradients