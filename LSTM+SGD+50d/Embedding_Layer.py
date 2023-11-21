# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:15:16 2023

@author: fist5
"""

import numpy as np
import emo_utils

class Embedding_Layer():
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.words_to_index, self.index_to_words, self.word_to_vec_map = emo_utils.read_glove_vecs(r'glove.6B.50d.txt')
    
    def forward(self, input_sentense):
        output = []
        for word in input_sentense:
            output.append(np.reshape(self.word_to_vec_map[word[0].lower()], (self.output_size, 1)))
        return output