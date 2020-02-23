#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class Highway(nn.Module):
    #pass
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f

    def __init__(self, word_embed_size):
        
        super(Highway, self).__init__()

        self.proj = nn.Linear(word_embed_size, word_embed_size, bias = True) #W_proj
        self.gate = nn.Linear(word_embed_size, word_embed_size, bias = True) #W_gate
        
        

    def forward(self, x_conv_out) -> torch.Tensor:
        """ Take a mini-batch of words, and compute highway output (word level) to feed to LSTM.
        """
        
        x_proj = torch.relu(self.proj(x_conv_out))
        x_gate = torch.sigmoid(self.gate(x_conv_out))
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out
        return x_highway
                 

    ### END YOUR CODE

# =============================================================================
# class HighwaySanityChecks(unittest.TestCase):
# 
#     def test_shape(self):
#         batch_size, word_embed_size = 64, 40
#         highway = Highway(word_embed_size, 0.5)
# 
#         x_conv_out = torch.randn([batch_size, word_embed_size])
#         x_word_emb = highway.forward(x_conv_out)
# 
#         self.assertEqual(x_word_emb.shape, (batch_size, word_embed_size))
#         self.assertEqual(x_word_emb.shape, x_conv_out.shape)
# 
#     def test_gate_bypass(self):
#         batch_size, word_embed_size = 64, 40
#         highway = Highway(word_embed_size, 0.0)
#         highway.gate.weight.data[:, :] = 0.0
#         highway.gate.bias.data[:] = -math.inf
# 
#         x_conv_out = torch.randn([batch_size, word_embed_size])
#         x_word_emb = highway.forward(x_conv_out)
# 
#         self.assertTrue(torch.allclose(x_conv_out, x_word_emb))
# 
#     def test_gate_projection(self):
#         batch_size, word_embed_size = 64, 40
#         highway = Highway(word_embed_size, 0.0)
#         highway.proj.weight.data = torch.eye(word_embed_size)
#         highway.proj.bias.data[:] = 0.0
#         highway.gate.weight.data[:, :] = 0.0
#         highway.gate.bias.data[:] = +math.inf
# 
#         x_conv_out = torch.rand([batch_size, word_embed_size])
#         x_word_emb = highway.forward(x_conv_out)
# 
#         self.assertTrue(torch.allclose(x_conv_out, x_word_emb))
# 
# 
# =============================================================================
