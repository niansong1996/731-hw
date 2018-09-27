# coding=utf-8

'''
    functions to generate embedding for each word and sentences
'''

from typing import List
from torch import Tensor
import vocab

# look_up the dict to convert to indices and do the padding
def corpus_to_indices(vocab: vocab.VocabEntry, corpus: List[List[str]]) -> Tensor:

    raise NotImplementedError()

# look up the dict for indices and convert to varied length sents
def indices_to_corpus(vocab: vocab.VocabEntry, indices: Tensor) -> List[List[str]]:

    raise NotImplementedError()
