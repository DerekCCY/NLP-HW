#!/usr/bin/env python
# -*- coding: utf-8 -*-
#   
# Copyright (C) 2024
# 
# @author: Ezra Fu <erzhengf@andrew.cmu.edu>
# based on work by 
# Ishita <igoyal@andrew.cmu.edu> 
# Suyash <schavan@andrew.cmu.edu>
# Abhishek <asrivas4@andrew.cmu.edu>

"""
11-411/611 NLP Assignment 2
RNN Language Model Implementation

Complete the LanguageModel class and other TO-DO methods.
"""

#######################################
# Import Statements
#######################################
from utils import *
from collections import Counter
from itertools import product
import argparse
import random
import math
import torch.nn.functional as F

#######################################
# TODO: RNNLanguageModel()
#######################################
class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, embedding_matrix):
        """
        RNN model class.
        
        Args
        ____
        vocab_size: int
            Size of the vocabulary
        embedding_dim: int
            Dimension of the word embeddings
        hidden_dim: int
            Dimension of the hidden state of the RNN
        embedding_matrix: torch.Tensor
            Pre-trained GloVe embeddings
            
        Other attributes:
            self.embedding: nn.Embedding
                Embedding layer
            self.rnn: nn.RNN
                RNN layer
            self.fc: nn.Linear
                Fully connected layer
        
        Note: Remember to initialize the weights of the embedding layer with the GloVe embeddings
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.embedding.weight = nn.Parameter(embedding_matrix)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        """
        The forward pass of the RNN model.
        
        Args
        ____
        x: torch.Tensor
            Input tensor of shape (batch_size, sequence_length)
        hidden: torch.Tensor
            Hidden state tensor of shape (num_layers, batch_size, hidden_dim)
        
        Returns
        -------
        out: torch.Tensor
            Output tensor of shape (batch_size, sequence_length, vocab_size)
        hidden: torch.Tensor
            Hidden state tensor of shape (num_layers, batch_size, hidden_dim)
            
        HINT: You need to use the embedding layer, rnn layer and the fully connected layer to define the forward pass
        """
        embedding_layer = self.embedding(x)
        out, hidden = self.rnn(embedding_layer, hidden)
        out = self.fc(out)
        return out, hidden

    
    def generate_sentence(self, sequence, word_to_ix, ix_to_word, num_words, mode='max'):
        """
        Predicts the next words given a sequence.
        
        Args
        ____
        sequence: str
            Input sequence
        word_to_ix: dict
            Dictionary mapping words to their corresponding indices
        ix_to_word: dict
            Dictionary mapping indices to their corresponding words
        num_words: int
            Maximum number of words to predict
        mode: str
            Mode of prediction. 'max' or 'multinomial'
            'max' mode selects the word with maximum probability
            'multinomial' mode samples the word from the probability distribution
            Hint: Use torch.multinomial() method
        
        Returns
        -------
        predicted_sequence: List[str]
            List of predicted words
        """
        self.eval()  # Set the model to evaluation mode
        sequence = sequence.split()  # Convert string to list of words
        predicted_sequence = []
        with torch.no_grad():
            input_seq = [word_to_ix.get(word, word_to_ix['<UNK>']) for word in sequence]
            input = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0)  # Shape: (1, sequence_length)
            #print(f"Input sequence: {input}")
            #print(f"Input Sequence shape: {input.shape}")
            hidden = None  # Initialize hidden state

            for _ in range(num_words):
                output, hidden = self.forward(input, hidden)
                #print(f"Output sequence: {output}")
                #print(f"Output Sequence shape: {output.shape}")

                # Get the output of the last time step
                output = output[:, -1, :]  # Shape: (1, vocab_size)

                if mode == 'max':
                    predicted_word_ix = torch.argmax(output, dim=1).item()
                elif mode == 'multinomial':
                    probabilities = torch.softmax(output, dim=1)
                    predicted_word_ix = torch.multinomial(probabilities, num_samples=1).item()
                else:
                    raise ValueError("Invalid mode. Choose 'max' or 'multinomial'.")

                predicted_word = ix_to_word.get(predicted_word_ix, '<UNK>')
                #print(f"Predicted word: {predicted_word}")
                predicted_sequence.append(predicted_word)

                # Prepare input for the next time step
                input = torch.tensor([[predicted_word_ix]], dtype=torch.long)
        return predicted_sequence