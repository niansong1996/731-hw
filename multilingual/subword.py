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
from config import LANG_NAMES


def train(lang, vocab_size):
    spm.SentencePieceTrainer. \
        Train('--pad_id=3 --character_coverage=1.0 --input=data/%s_mono.txt --model_prefix=subword_files/%s --vocab_size=%d' % (lang, lang, vocab_size))


def get_corpus_pairs(src_lang_idx: int, tgt_lang_idx: int, data_type: str) \
        -> List[Tuple[List[int], List[int]]]:
    # get src and tgt corpus ids separately
    src_sents, long_sent = get_corpus_ids(src_lang_idx, tgt_lang_idx, data_type, False)
    tgt_sents, _ = get_corpus_ids(src_lang_idx, tgt_lang_idx, data_type, True, long_sent=long_sent)

    # pair those corresponding sents together
    src_tgt_sent_pairs = list(zip(src_sents, tgt_sents))

    max_length = max([len(x) for x in src_sents])
    print('lang pair %s-%s before %d after %d max src length %d' % (LANG_NAMES[src_lang_idx], LANG_NAMES[tgt_lang_idx], \
          len(src_sents) + len(long_sent), len(src_tgt_sent_pairs), max_length))

    return src_tgt_sent_pairs


def get_corpus_ids(src_lang_idx: int, tgt_lang_idx: int, data_type: str, is_tgt: bool, is_train=True, long_sent=None)\
        -> Tuple[List[List[int]], Set[int]]:
    sents = []

    src_lang = LANG_NAMES[src_lang_idx]
    tgt_lang = LANG_NAMES[tgt_lang_idx]
    lang = tgt_lang if is_tgt else src_lang

    # load the subword models for encoding these sents to indices
    sp = spm.SentencePieceProcessor()
    sp.Load('subword_files/%s.model' % lang)

    # read corpus for corpus
    file_path = 'data/%s.%s-%s.%s.txt' % (data_type, src_lang, tgt_lang, lang)
    line_count = 0
    long_sent_in_src = set()
    for line in open(file_path, encoding="utf-8"):
        sent = line.strip()
        line_count += 1
        if is_tgt:
            if line_count in long_sent:
                continue
        else:
            if is_train and len(sent.split(' ')) > 50:
                long_sent_in_src.add(line_count)
                continue

        if src_lang_idx == tgt_lang_idx and not is_tgt:
            # denoising autoencoder
            sent_words = sent.split(' ')
            np.random.shuffle(sent_words)
            sent = ' '.join(sent_words)

        # convert to subword ids
        sent_encode = []
        for word in sent.split(' '):
            word_encode = sp.EncodeAsIds(word)
            if len(word_encode) > 8 and len(word) < 2*len(word_encode):
                sent_encode += [sp.unk_id()]
            else:
                sent_encode += word_encode

        if not is_tgt and is_train and len(sent_encode) > 75:
            long_sent_in_src.add(line_count)
            continue

        if is_tgt:
            # add <s> and </s> to the tgt sents
            sent_encode = [sp.bos_id()] + sent_encode + [sp.eos_id()]

        sents.append(sent_encode)
    return sents, long_sent_in_src


def decode_corpus_ids(lang_name: str, sents: List[List[int]]) -> List[List[str]]:
    sp = spm.SentencePieceProcessor()
    sp.Load('subword_files/%s.model' % lang_name)

    decoded_sents = []
    for line in sents:
        sent = sp.DecodeIds(line)
        decoded_sents.append(sent)

    return decoded_sents


if __name__ == '__main__':
    args = docopt(__doc__)

    vocab_size = int(args['--vocab-size'])
    lang = args['--lang']

    print('building subword model for %s language : ' % lang)

    # train the subword model for the specified language
    if lang == 'all':
        for lan in LANG_NAMES.values():
            train(lan, vocab_size)
            print('Done for %s : ' % lang)
    else:
        train(lang, vocab_size)
        print('Done for %s : ' % lang)
