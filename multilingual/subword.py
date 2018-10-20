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
from typing import List, Tuple

import sentencepiece as spm
from docopt import docopt
from config import LANG_NAMES


def train(lang, vocab_size):
    spm.SentencePieceTrainer.\
    Train('--input=data/%s_mono.txt --model_prefix=subword_files/%s --vocab_size=%d' % (lang, lang, vocab_size))


def get_corpus_pairs(src_lang_idx: int, tgt_lang_idx: int, data_type: str) \
        -> List[Tuple[List[int], List[int]]]:
    # get src and tgt corpus ids separately
    src_sents = get_corpus_ids(src_lang_idx, tgt_lang_idx, data_type, False)
    tgt_sents = get_corpus_ids(src_lang_idx, tgt_lang_idx, data_type, True)

    # pair those corresponding sents together
    src_tgt_sent_pairs = list(zip(src_sents, tgt_sents))

    return src_tgt_sent_pairs

def get_corpus_ids(src_lang_idx: int, tgt_lang_idx: int, data_type: str, is_tgt: bool):
    sents = []

    src_lang = LANG_NAMES[src_lang_idx]
    tgt_lang = LANG_NAMES[tgt_lang_idx]
    lang = tgt_lang if is_tgt else src_lang

    # load the subword models for encoding these sents to indices
    sp = spm.SentencePieceProcessor()
    sp.Load('subword_files/%s.model' % lang)

    # read corpus for corpus
    file_path = 'data/%s.%s-%s.%s.txt' % (data_type, tgt_lang, src_lang, lang)
    for line in open(file_path, encoding="utf-8"):
        sent = line.strip()

        sent_encode = src_sp.EncodeAsIds(sent)
        if is_tgt:
            # add <s> and </s> to the tgt sents
            sent_encode = [tgt_sp.bos_id()] + sent_encode + [tgt_sp.eos_id()]
        sents.append(sent_encode)

    return sents

def decode_corpus_ids(lang: int, sents: List[List[int]]) -> List[List[str]]:
    sp = spm.SentencePieceProcessor()
    sp.Load('subword_files/%s.model' % lang)

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