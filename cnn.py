#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import unittest

class CNN(nn.Module):
    #pass
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g

    def __init__(self, char_embed_size, word_embed_size, kernel_size: int=5, pad_size: int=1):
        
        super(CNN, self).__init__()
        
        #word_embed_size = num of filters (as specified in question) = num output channels
        self.conv = nn.Conv1d(char_embed_size, word_embed_size, kernel_size=kernel_size, padding=pad_size, bias = True) #W
        

    def forward(self, x) -> torch.Tensor:
        # input tensor size should be (batch_size, char_embed_size, max_word_len = number of characters in one word) 

        x_conv = self.conv(x)
        x_conv_out, x_conv_idx = torch.max(torch.relu(x_conv), dim=2)
        return x_conv_out


    ### END YOUR CODE

class CNNSanityChecks():
    #red box from publication

    def test_shape(self):
        max_word_len = 9
        batch_size, char_embed_size, word_embed_size = 10, 4, 5
        cnn = CNN(char_embed_size, word_embed_size)

        #input is 10 x 4 x 9
        x_emb = torch.randn([batch_size, char_embed_size, max_word_len])
        
        #cnn input channels 4, output 5 -> since 10 batches, expected size is 10x5
        x_conv_out = cnn.forward(x_emb)

        assert x_conv_out.shape == (batch_size, word_embed_size), "output shape does not match"
        
        print("Sanity Check Passed for Question 1e: To Input Tensor Char!")