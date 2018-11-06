#!/usr/bin/env python
"""
Generate the vocabulary file for neural network training
A vocabulary file is a mapping of tokens to their indices

Usage:
    vocab.py --lower-case=<BOOL> [options] SRC_FILE

Options:
    -h --help                  Show this screen.
    --freq-size=<int>          vocab size for most frequent words [default: 5000]
    --lower-case=BOOL
    --min-len=<int>
    --max-len=<int>
    --max-size=<int>
    --unk-size=<int>           vocab size for calculation non-unk words [default: 50000]
    --freq-cutoff=<int>        frequency cutoff [default: 2]
"""

from collections import Counter
from itertools import chain
from docopt import docopt


def conditional_lower_case(sent: str, all_lower_case: bool):
    if all_lower_case:
        return sent.lower()
    else:
        # lower the case of the first word only when first char is only char that upper
        first_word = sent.split(' ')[0]
        first_word = first_word[0].lower() + first_word[1:]
        if first_word.islower():
            return sent[0].lower() + sent[1:]
    return sent


def read_corpus(file_path, all_lower_case: bool):
    data = []
    for line in open(file_path, encoding="utf-8"):
        stripped_line = line.strip()
        sub_sents = stripped_line.split('.')
        for i, sub_sent in enumerate(sub_sents):
            sub_sent = sub_sent.strip()
            if len(sub_sent) == 0:
                continue
            if sub_sent[0].islower():
                if len(data) != 0:
                    sub_sent_words = sub_sent.strip().split(' ')
                    if i != len(sub_sents) - 1 or stripped_line[-1] == '.':
                        sub_sent_words += '.'
                    data[-1] += sub_sent_words
                    continue
            else:
                sub_sent = conditional_lower_case(sub_sent, all_lower_case)
                sub_sent_words = sub_sent.strip().split(' ')
                if i != len(sub_sents) - 1 or stripped_line[-1] == '.':
                    sub_sent_words += '.'
                data.append(sub_sent_words)

    return data


class VocabEntry(object):
    def __init__(self):
        self.unk_id = -1

        self.word2id = dict()
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
    def from_corpus(corpus, vocab_file_path, freq_size, unk_size, freq_cutoff=2):
        vocab_entry = VocabEntry()

        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print(
            f'number of word types: {len(word_freq)}, number of word types w/ frequency >= {freq_cutoff}: {len(valid_words)}')

        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:unk_size]
        with open(vocab_file_path, 'w', encoding='utf-8') as vocab_file:
            for i in range(unk_size):
                vocab_entry.add(top_k_words[i])
                if i < freq_size:
                    vocab_file.write(top_k_words[i] + '\n')

        return vocab_entry


def preprocess_wiki(sents, min_length=5, max_length=40, max_size=2000000):
    processed_sents = []

    # screen the sents with the following standards
    for sent in sents:
        if len(sent) < min_length or len(sent) > max_length:
            continue
        # unk_num = 0
        # for i in range(len(sent)):
        #     if sent[i] == '.' and not i == (len(sent) - 1):
        #         sent[i] = '.\n'
        # if unk_num < 0.2 * len(sent):
        #     processed_sents.append(sent)
        processed_sents.append(sent)

        # when max sents num is reached
        if len(processed_sents) == max_size:
            break
    return processed_sents


if __name__ == '__main__':
    args = docopt(__doc__)

    corpus_file = args['SRC_FILE']
    min_length = int(args['--min-len'])
    max_length = int(args['--max-len'])
    max_size = int(args['--max-size'])

    # vocab_file = args['VOCAB_FILE']
    # freq_size = int(args['--freq-size'])
    # unk_size = int(args['--unk-size'])
    # freq_cutoff = int(args['--freq-cutoff'])

    src_sents = read_corpus(corpus_file, args['--lower-case'] == 'True')
    # vocab = VocabEntry.from_corpus(src_sents, vocab_file, freq_size, unk_size, freq_cutoff)
    # processed_wiki = preprocess_wiki(src_sents, 5, 40, 500000, vocab)
    processed_wiki = preprocess_wiki(src_sents, min_length=min_length, max_length=max_length, max_size=max_size)
    # save the processed wiki 
    file_name = corpus_file + '.prep'
    with open(file_name, 'w', encoding='utf-8') as out_file:
        for sent in processed_wiki:
            out_file.write((' '.join(sent)).strip() + '\n')
    print('preprocessed wiki is saved to %s' % file_name)
