#!/usr/bin/env python
"""
Generate the vocabulary file for neural network training
A vocabulary file is a mapping of tokens to their indices

Usage:
    vocab.py --train-src=<file> [options] VOCAB_FILE

Options:
    -h --help                  Show this screen.
    --train-src=<file>         File of training source sentences
    --size=<int>               vocab size [default: 50000]
    --freq-cutoff=<int>        frequency cutoff [default: 2]
"""

from typing import List
from collections import Counter
from itertools import chain
from docopt import docopt
import pickle


class VocabEntry(object):
    def __init__(self, vocab_size):
        self.word2id = dict()
        self.unk_id = Vocab.UNK_ID
        self.word2id[Vocab.PAD] = Vocab.PAD_ID
        self.word2id[Vocab.SOS] = Vocab.SOS_ID
        self.word2id[Vocab.EOS] = Vocab.EOS_ID
        self.word2id[Vocab.UNK] = Vocab.UNK_ID
        self.vocab_size = vocab_size

        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self.id2word[wid]

    def add(self, word):
        if len(self) >= self.vocab_size:
            return -1
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    @staticmethod
    def from_corpus(corpus, vocab_size, freq_cutoff=2):
        vocab_entry = VocabEntry(vocab_size)

        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print(f'number of word types: {len(word_freq)}, number of word types w/ frequency >= {freq_cutoff}: {len(valid_words)}')

        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:vocab_size]
        for word in top_k_words:
            vocab_entry.add(word)

        return vocab_entry


class Vocab(object):
    UNK = '<unk>'
    SOS = '<s>'
    EOS = '</s>'
    PAD = '<pad>'
    UNK_ID = 0
    SOS_ID = 1
    EOS_ID = 2
    PAD_ID = 3

    def __init__(self, sents, vocab_size, freq_cutoff):
        self.vocab_entry = VocabEntry.from_corpus(sents, vocab_size, freq_cutoff)

    def __repr__(self):
        return 'Vocab %d words' % len(self.vocab_entry)


if __name__ == '__main__':
    args = docopt(__doc__)

    print('read in source sentences: %s' % args['--train-src'])
    sents = []
    for line in open(args['--train-src'], encoding="utf-8"):
        sent = line.strip().split(' ')
        sents.append(sent)
    vocab = Vocab(sents, int(args['--size']), int(args['--freq-cutoff']))
    print('generated vocabulary %d words' % len(vocab.vocab_entry))
    pickle.dump(vocab, open(args['VOCAB_FILE'], 'wb'))
    print('vocabulary saved to %s' % args['VOCAB_FILE'])
