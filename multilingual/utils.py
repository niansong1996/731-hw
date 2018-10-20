import math
from typing import List, Tuple

import torch

from config import LANG_NAMES
from collections import namedtuple

import numpy as np
import io
import torch.tensor as Tensor

from vocab import Vocab

LangPair = namedtuple('LangPair', ['src', 'tgt'])
PairedData = namedtuple('PairedData', ['data', 'langs'])


def input_transpose(sents, pad_token):
    """
    This function transforms a list of sentences of shape (batch_size, token_num) into 
    a list of shape (token_num, batch_size). You may find this function useful if you
    use pytorch
    """
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])

    return sents_t


def read_corpus(file_path, source):
    data = []
    for line in open(file_path, encoding="utf-8"):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def assert_tensor_size(tensor: Tensor, expected_size: List[int]):
    try:
        assert list(tensor.shape) == expected_size
    except AssertionError:
        print("tensor shape %s doesn't match expected size %s" % (tensor.shape, expected_size))
        raise


def batch_iter(data: List[PairedData], batch_size, shuffle=True) -> Tuple[int, int, List[List[int]], List[List[int]]]:
    """
    Given a list of examples, shuffle and slice them into mini-batches
    """
    pairs = [PairedDataBatch(i, pd, batch_size, shuffle) for i, pd in enumerate(data)]
    batch_indices = [batch_idx for p in pairs for batch_idx in p.batch_indices]
    if shuffle:
        np.random.shuffle(batch_indices)
    for pair_idx, batch_idx in batch_indices:
        pair = pairs[pair_idx]
        src_sents, tgt_sents = pair.get_batch(batch_idx)
        yield pair.src_lang, pair.tgt_lang, src_sents, tgt_sents


class PairedDataBatch:
    def __init__(self, pair_idx: int, paried_data: PairedData, batch_size, shuffle=True):
        self.data = paried_data.data
        self.src_lang = paried_data.langs.src
        self.tgt_lang = paried_data.langs.tgt
        self.batch_size = batch_size
        batch_count = math.ceil(len(self.data) / batch_size)
        self.index_array = list(range(len(self.data)))

        # sort the pairs w.r.t. the length of the src sent
        self.data = sorted(self.data, key=lambda e: len(e[0]), reverse=True)

        self.batch_indices = [(pair_idx, i) for i in range(batch_count)]
        if shuffle:
            np.random.shuffle(self.batch_indices)
        self.i = 0

    def get_batch(self, batch_idx: int) -> Tuple[List[List[int]], List[List[int]]]:
        indices = self.index_array[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
        examples = [self.data[idx] for idx in indices]

        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]
        yield src_sents, tgt_sents


def sents_to_tensor(sents: List[List[int]], device: torch.device) -> Tensor:
    max_sent_len = max(map((lambda x: len(x)), sents))
    # indices are initialized with the index of '<pad>'
    for sent in enumerate(sents):
        while len(sent) < max_sent_len:
            sent.append(Vocab.PAD_ID)
    return torch.tensor(sents, dtype=torch.long, device=device)


def load_matrix(fname, vocabs, emb_dim):
    words = []
    word2idx = {}
    word2vec = {}

    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        word2idx[word] = len(words)
        words.append(word)
        word2vec[word] = np.array(tokens[1:]).astype(np.float)

    matrix_len = len(vocabs)
    weights_matrix = np.zeros((matrix_len, emb_dim))
    words_found = 0

    for i, word in enumerate(vocabs):
        try:
            weights_matrix[i] = word2vec[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.random(size=(emb_dim,))
    return weights_matrix
