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
N-gram Language Model Implementation

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

#######################################
# TODO: get_ngrams()
#######################################
def get_ngrams(list_of_words, n):
    """
    Returns a list of n-grams for a list of words.
    Args
    ----
    list_of_words: List[str]
        List of already preprocessed and flattened (1D) list of tokens e.g. ["<s>", "hello", "</s>", "<s>", "bye", "</s>"]
    n: int
        n-gram order e.g. 1, 2, 3
    
    Returns:
        n_grams_list: List[Tuple]
            Returns a list containing n-gram tuples
    """
    n_grams_list = []
    for words in range(len(list_of_words)):
        n_grams_list.append(tuple(list_of_words[words:words+n]))
    if n==1:
        return n_grams_list
    return n_grams_list[:-(n-1)]

#######################################
# TODO: NGramLanguageModel()
#######################################
class NGramLanguageModel():
    def __init__(self, n, train_data, alpha=1):
        """
        Language model class.
        
        Args
        ____
        n: int
            n-gram order
        train_data: List[List]
            already preprocessed unflattened list of sentences. e.g. [["<s>", "hello", "my", "</s>"], ["<s>", "hi", "there", "</s>"]]
        alpha: float
            Smoothing parameter
        
        Other attributes:
            self.tokens: list of individual tokens present in the training corpus
            self.vocab: vocabulary dict with counts
            self.model: n-gram language model, i.e., n-gram dict with probabilties
            self.n_grams_counts: dictionary for storing the frequency of ngrams in the training data, keys being the tuple of words(n-grams) and value being their frequency
            self.prefix_counts: dictionary for storing the frequency of the (n-1) grams in the data, similar to the self.n_grams_counts
            As an example:
            For a trigram model, the n-gram would be (w1,w2,w3), the corresponding [n-1] gram would be (w1,w2)
        """
        self.n = n 
        self.train_data = train_data
        self.alpha = alpha
        self.tokens = flatten(self.train_data)
        self.vocab = Counter(self.tokens)
        # Initialize the n_grams_counts and prefix_counts before setting up the model
        self.n_grams_counts = {}
        self.prefix_counts = {}
        for ngram in get_ngrams(self.tokens, self.n):
            self.n_grams_counts[ngram] = self.n_grams_counts.get(ngram,0) + 1
            if self.n > 1:
                prefix = ngram[:-1]
                self.prefix_counts[prefix] = self.prefix_counts.get(prefix,0) + 1
        
        # Now initialize the model using the n_grams_counts
        self.model = {}
        for ngram in get_ngrams(self.tokens, self.n):
            self.model[ngram] = self.get_prob(ngram)


    def build(self):
        """
        Returns a n-gram dict with their smoothed probabilities. Remember to consider the edge case of n=1 as well
        
        You are expected to update the self.n_grams_counts and self.prefix_counts, and use those calculate the probabilities. 
        """
        if not self.tokens:
            print("Warning: No tokens found in training data.")
            return {}

        for ngram in get_ngrams(self.tokens, self.n):
            self.model[ngram] = self.get_prob(ngram)
        return self.model
            
        #for i in range(len(self.tokens) - self.n + 1):
        #    ngram = tuple(self.tokens[i:i+self.n])
        #    #print(f"Processing ngram: {ngram}")
        #    self.n_grams_counts[ngram] = self.n_grams_counts.get(ngram,0) + 1
#
        #    if self.n > 1:
        #        prefix = tuple(self.tokens[i:i+self.n-1])
        #        self.prefix_counts[prefix] = self.prefix_counts.get(prefix,0) + 1
        #
        #for ngram, count in self.n_grams_counts.items():
        #    if self.n == 1:
        #        self.model[ngram] = (count + self.alpha) / (len(self.tokens) + self.alpha * len(self.vocab))
        #    else:
        #        prefix = ngram[:-1]
        #        self.model[ngram] = (count + self.alpha) / (self.prefix_counts[prefix] + self.alpha * len(self.vocab))
    #
        #return self.model

    def get_smooth_probabilities(self, ngrams):
        """
        Returns the smoothed probability of the n-gram, using Laplace Smoothing. 
        Remember to consider the edge case of  n = 1
        HINT: Use self.n_gram_counts, self.tokens and self.prefix_counts 
        """
        #print(ngrams)
        ngrams_counts = self.n_grams_counts.get(ngrams,0)
        #print(ngrams_counts)
        if self.n > 1:
            prefix_counts = self.prefix_counts.get(ngrams[:-1],0)
            #print(f"Prefix Counts: {prefix_counts}")
            smooth_probabilities = (ngrams_counts + self.alpha) / (prefix_counts + self.alpha * len(self.vocab))
        elif self.n == 1:
            smooth_probabilities = (ngrams_counts + self.alpha) / (len(self.tokens) + self.alpha * len(self.vocab))
        return smooth_probabilities

    def get_prob(self, ngram):
        """
        Returns the probability of the n-gram, using Laplace Smoothing.
        
        Args
        ____
        ngram: tuple
            n-gram tuple
        
        Returns
        _______
        float
            probability of the n-gram
        """
        probability = self.model.get(ngram,0)
        if probability == 0:
            # If the n-gram is not observed in the training data, use smoothed probabilities
            probability = self.get_smooth_probabilities(ngram)

        return probability



    def perplexity(self, test_data):
        """
        Returns perplexity calculated on the test data.
        Args
        ----------
        test_data: List[List] 
            Already preprocessed nested list of sentences

        Returns
        -------
        float
            Calculated perplexity value
        """
        # Flatten the test data
        flattened_test_data = flatten(test_data)
        test_ngrams = get_ngrams(flattened_test_data, self.n)
        N = len(test_ngrams)
        log_prob_sum = 0

        for ngrams in test_ngrams:
            prob = self.get_prob(ngrams)
            log_prob_sum += math.log(prob)

        perplexity = math.exp(-log_prob_sum/N)
        # Return perplexity using exponentiation of the negative average log probability
        return perplexity 
    
###############################################
# Method: Most Probable Candidates [Don't Edit]
###############################################

# Copy of main executable script provided locally for your convenience
# This is not executed on autograder, so do what you want with it
if __name__ == '__main__':
    train = "data/sample.txt"
    test = "data/sample.txt"
    n = 2
    alpha = 0.1

    print("No of sentences in train file: {}".format(len(train)))
    print("No of sentences in test file: {}".format(len(test)))

    print("Raw train example: {}".format(train[2]))
    print("Raw test example: {}".format(test[2]))

    train = preprocess(train, n)
    test = preprocess(test, n)

    print("Preprocessed train example: \n{}\n".format(train[2]))
    print("Preprocessed test example: \n{}".format(test[2]))

    # Language Model
    print("Loading {}-gram model.".format(n))
    lm = NGramLanguageModel(n, train, alpha)
    lm.build()

    print("Vocabulary size (unique unigrams): {}".format(len(lm.vocab)))
    print("Total number of unique n-grams: {}".format(len(lm.model)))
    
    # Perplexity
    ppl = lm.perplexity(train)
    print("Model perplexity: {:.3f}".format(ppl))
    
    s1 = ("Every", "time", "you",'come')
    # Generating sentences using your model
    print("Generating random sentences.")
    sentences = list(generate_sentences_from_phrase(lm, 1, list(s1), 1, 'max'))
    print(sentences)    