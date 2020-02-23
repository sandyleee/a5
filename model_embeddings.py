#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>'] # notice that in assignment 4 vocab is of type (Vocab), not (VocabEntry) as assignment 5.
        # self.embeddings = nn.Embedding(len(vocab.src), word_embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1h
        pad_token_idx = vocab.char_pad
        char_embed_size = 50
        dropout_prob = 0.3
        self.word_embed_size = word_embed_size
        
        self.embeddings = nn.Embedding(len(vocab.char2id), char_embed_size, padding_idx=pad_token_idx)
        self.cnn = CNN(char_embed_size, word_embed_size)
        self.highway = Highway(word_embed_size)
        
        self.dropout = nn.Dropout(dropout_prob)
        
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1h
        x_emb = self.embeddings(input)     
        sentence_length, batch_size, max_word_length, char_embed_size = x_emb.shape
        
        x_conv = x_emb.view(sentence_length*batch_size, max_word_length, char_embed_size).permute(0, 2, 1)
        # (sentence_length * batch_size, char_embed_size, max_word_length) <- after reshape + permute
        
        x_conv_out = self.cnn(x_conv)
        # becomes (sentence_length * batch_size, word_embed_size)
        
        x_highway = self.highway(x_conv_out)
        # still (sentence_length * batch_size, word_embed_size)

        x_word_emb = self.dropout(x_highway)
        # still (sentence_length * batch_size, word_embed_size)
         
        output = x_word_emb.view(sentence_length, batch_size, -1)
        # (sentence_length, batch_size, word_embed_size)
        
        return output

        ### END YOUR CODE

