#!/usr/bin/env python
"""
Generate the subword models and vocab for languages 
The model and vocab can be further used to encode and decode using provided functions

Usage:
    vocab.py --lang=<lang-abbr> --vocab-size=<file> 

Options:
    -h --help                  Show this screen.
    --lang=<lang-abbr>         Two letter representation of language
    --vocab-size=<file>        The vocabulary size for subword model
"""
from typing import List, Tuple, Set

import sentencepiece as spm
import numpy as np
from docopt import docopt
from config import LANG_NAMES, LANG_INDICES
from vocab import Vocab
from wiki_prep import conditional_lower_case


def train(lang, vocab_size):
    spm.SentencePieceTrainer. \
        Train('--pad_id=3 --character_coverage=1.0 --input=data/%s_mono.txt --model_prefix=subword_files/%s --vocab_size=%d' % (lang, lang, vocab_size))


def get_corpus_pairs(vocabs, src_lang_idx: int, tgt_lang_idx: int, data_type: str) \
        -> List[Tuple[List[int], List[int]]]:
    # get src and tgt corpus ids separately
    src_sents, long_sent = get_corpus_ids(vocabs, src_lang_idx, tgt_lang_idx, data_type, False)
    tgt_sents, _ = get_corpus_ids(vocabs, src_lang_idx, tgt_lang_idx, data_type, True, long_sent=long_sent)
    print("original # sents: %d" % len(src_sents))
    # pair those corresponding sents together
    src_tgt_sent_pairs = list(filter(lambda p: len(p[0]) / len(p[1]) > 0.3 or
                                               len(p[1]) < 8,
                                     zip(src_sents, tgt_sents)))
    print("filtered # sents: %d" % len(src_tgt_sent_pairs))
    return src_tgt_sent_pairs


def get_corpus_ids(vocabs, src_lang_idx: int, tgt_lang_idx: int, data_type: str, is_tgt: bool,
                   is_train=True, long_sent=None) -> Tuple[List[List[int]], Set[int]]:
    sents = []
    lang_idx = tgt_lang_idx if is_tgt else src_lang_idx
    src_lang = LANG_NAMES[src_lang_idx]
    tgt_lang = LANG_NAMES[tgt_lang_idx]
    lang = LANG_NAMES[lang_idx]
    vocab = vocabs[lang_idx]

    # read corpus for corpus
    file_path = 'data/%s.%s-%s.%s.txt' % (data_type, src_lang, tgt_lang, lang)
    line_count = 0
    long_sent_in_src = set()
    for line in open(file_path, encoding="utf-8"):
        sent = line.strip()
        sent = conditional_lower_case(sent, not is_tgt)
        line_count += 1
        if is_tgt:
            if line_count in long_sent:
                continue
        else:
            if is_train and len(sent.split(' ')) > 50:
                long_sent_in_src.add(line_count)
                continue
        sent_words = sent.split(' ')
        if src_lang_idx == tgt_lang_idx and not is_tgt:
            # denoising autoencoder
            np.random.shuffle(sent_words)

        # convert to subword ids
        sent_encode = vocab.words2indices(sent_words)
        if is_tgt:
            # add <s> and </s> to the tgt sents
            sent_encode = [Vocab.SOS_ID] + sent_encode + [Vocab.EOS_ID]
        sents.append(sent_encode)
    return sents, long_sent_in_src


def decode_corpus_ids(vocabs, lang_name: str, sents: List[List[int]]) -> List[str]:
    vocab = vocabs[LANG_INDICES[lang_name]]
    decoded_sents = []
    for id_list in sents:
        try:
            end = id_list.index(Vocab.EOS_ID) - 1
        except ValueError:
            end = len(id_list)
        word_list = [vocab.id2word[w] for w in id_list[1:end]]  # skip <s>, stop at </s>
        if len(word_list) > 0:
            word_list[0] = word_list[0][0].upper() + word_list[0][1:]
        i = 1
        while i < len(word_list):
            if word_list[i - 1] == '.':
                word_list[i] = word_list[i][0].upper() + word_list[i][1:]
            i += 1
        decoded_sents.append(' '.join(word_list))
    return decoded_sents


if __name__ == '__main__':
    args = docopt(__doc__)

    vocab_size = int(args['--vocab-size'])
    lang = args['--lang']

    print('building subword model for %s language : ' % lang)

    # train the subword model for the specified language
    if lang == 'all':
        for lan in LANG_NAMES:
            train(lan, vocab_size)
            print('Done for %s : ' % lang)
    else:
        train(lang, vocab_size)
        print('Done for %s : ' % lang)
