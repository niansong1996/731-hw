# coding=utf-8

'''
    functions to generate embedding for each word and sentences
'''

from functools import reduce
from typing import List
from torch import Tensor
import torch
import numpy as np
import vocab

# look_up the dict to convert to indices and do the padding
def corpus_to_indices(vocab: vocab.VocabEntry, corpus: List[List[str]]) -> Tensor:
    max_sent_len = max(map((lambda x: len(x)), corpus))
    # indices are initialized with the index of '<pad>'
    for i, sent in enumerate(corpus):
        while len(sent) < max_sent_len:
            sent.append("<pad>")

    indices_in_lists = vocab.words2indices(corpus)
    return torch.tensor(indices_in_lists, dtype=torch.long)
    '''
    indices = torch.zeros(len(corpus), max_sent_len, dtype=torch.int32)
    for i, sent in enumerate(corpus):
        sent_indices = np.zeros((1, max_sent_len))
        for j, word in enumerate(sent):
            sent_indices[j] = vocab.word2id[word]
        indices[i] = torch.tensor(sent_indices)
    return indices
    '''

# look up the dict for indices and convert to varied length sents
def indices_to_corpus(vocab: vocab.VocabEntry, indices: Tensor) -> List[List[str]]:
    corpus = []
    for i in range(indices.shape[0]):
        sent = []
        for j in range(indices.shape[1]):
            idx = indices[i][j]
            if idx == 0:
                break
            sent.append(vocab.id2word[idx])
        corpus.append(sent)
    return corpus
