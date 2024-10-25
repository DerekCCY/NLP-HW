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
This file contains the preprocessing and read() functions. Don't edit this file.
"""

from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import random
import math

START = "<s>"
EOS = "</s>"
UNK = "<UNK>"

def read_file(file_path):
    """
    Read a single text file.
    """
    with open(file_path, 'r') as f:
        text = f.readlines()
    return text

def preprocess(sentences, n):
    """
    Args:
        sentences: List of sentences
        n: n-gram value

    Returns:
        preprocessed_sentences: List of preprocessed sentences
    """
    sentences = add_special_tokens(sentences, n)

    preprocessed_sentences = []
    for line in sentences:
        preprocessed_sentences.append([tok.lower() for tok in line.split()])
    
    return preprocessed_sentences

def add_special_tokens(sentences, ngram):
    num_of_start_tokens = ngram - 1 if ngram > 1 else 1
    start_tokens = " ".join([START] * num_of_start_tokens)
    sentences = ['{} {} {}'.format(start_tokens, sent, EOS) for sent in sentences]
    return sentences


def flatten(lst):
    """
    Flattens a nested list into a 1D list.
    Args:
        lst: Nested list (2D)
    
    Returns:
        Flattened 1-D list
    """
    return [b for a in lst for b in a]

def loadfile(filename):
    # Read the file
    tokens = flatten(preprocess(read_file(filename), n=1))

    # Build a vocabulary: set of unique words
    vocab = set(tokens)
    vocab.add(START)
    vocab.add(EOS)
    vocab.add(UNK)
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    ix_to_word = {i: word for i, word in enumerate(vocab)}

    # Convert words to indices
    indices = [word_to_ix[word] for word in tokens]

    # Prepare dataset
    class WordDataset(Dataset):
        def __init__(self, data, sequence_length=30):
            self.data = data
            self.sequence_length = sequence_length

        def __len__(self):
            return len(self.data) - self.sequence_length

        def __getitem__(self, index):
            return (
                torch.tensor(self.data[index:index+self.sequence_length], dtype=torch.long),   # Input sequence
                torch.tensor(self.data[index+1:index+self.sequence_length+1], dtype=torch.long), # Next word
            )

    dataset = WordDataset(indices, sequence_length=min(30, len(indices)-1))
    return vocab, word_to_ix, ix_to_word, DataLoader(dataset, batch_size=2, shuffle=True)

def best_candidate(lm, prev, i, without=[], mode="random"):
    """
    Returns the most probable word candidate after a given sentence.
    """
    blacklist  = [UNK] + without
    candidates = ((ngram[-1],prob) for ngram,prob in lm.model.items() if ngram[:-1]==prev)
    candidates = filter(lambda candidate: candidate[0] not in blacklist, candidates)
    candidates = sorted(candidates, key=lambda candidate: candidate[1], reverse=True)
    if len(candidates) == 0:
        return ("</s>", 1)
    else:
        if(mode=="random"):
            return candidates[random.randrange(len(candidates))]
        else:
            return candidates[0]
        
def top_k_best_candidates(lm, prev, k, without=[]):
    """
    Returns the K most-probable word candidate after a given n-1 gram.
    Args
    ----
    lm: LanguageModel class object
    
    prev: n-1 gram
        List of tokens n
    """
    blacklist  = [UNK] + without
    candidates = ((ngram[-1],prob) for ngram,prob in lm.model.items() if ngram[:-1]==prev)
    candidates = filter(lambda candidate: candidate[0] not in blacklist, candidates)
    candidates = sorted(candidates, key=lambda candidate: candidate[1], reverse=True)

    if len(candidates) == 0:
        return ("</s>", 1)
    else:
        return candidates[:k]
    
def generate_sentences_from_phrase(lm, num, sent, prob, mode):
    """
    Generate sentences using the trained language model.
    """
    min_len=12
    max_len=24
    
    for i in range(num):
        while sent[-1] != "</s>":
            prev = () if lm.n == 1 else tuple(sent[-(lm.n-1):])
            blacklist = sent + (["</s>"] if len(sent) < min_len else [])

            next_token, next_prob = best_candidate(lm, prev, i, without=blacklist, mode=mode)
            sent.append(next_token)
            prob *= next_prob
            
            if len(sent) >= max_len:
                sent.append("</s>")

        yield ' '.join(sent), -1/math.log(prob)
        
def load_glove_embeddings(path):
    embeddings_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = torch.tensor([float(val) for val in values[1:]], dtype=torch.float)
            embeddings_dict[word] = vector
    return embeddings_dict

def create_embedding_matrix(word_to_ix, embeddings_dict, embedding_dim):
    vocab_size = len(word_to_ix)
    embedding_matrix = torch.zeros((vocab_size, embedding_dim))
    for word, ix in word_to_ix.items():
        if word in embeddings_dict:
            embedding_matrix[ix] = embeddings_dict[word]
        else:
            embedding_matrix[ix] = torch.rand(embedding_dim)  # Random initialization for words not in GloVe
    return embedding_matrix
