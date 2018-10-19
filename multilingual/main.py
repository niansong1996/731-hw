import math
import pickle
import sys
import time
from collections import namedtuple

import numpy as np
from typing import *
from docopt import docopt
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from CPG import CPG
from Decoder import Decoder
from utils import read_corpus, batch_iter, load_matrix
from vocab import Vocab, VocabEntry
from embed import corpus_to_indices, indices_to_corpus
from config import device, Hypothesis, LANG_INDICES

import torch
import torch.nn as nn
import torch.tensor as Tensor
import torch.nn.functional as F


def train(args: Dict[str, str]):
    train_corpus_pairs = [read_corpus(0, i, source='train') for i in range(1, len(LANG_INDICES))]
    dev_corpus_pairs = [read_corpus(0, i, source='dev') for i in range(1, len(LANG_INDICES))]

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']
    optimizer_save_path = args['--save-opt']

    vocab = pickle.load(open(args['--vocab'], 'rb'))

    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                vocab=vocab).to(device)

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cumulative_tgt_words = report_tgt_words = 0
    cumulative_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    # set the optimizers
    lr = float(args['--lr'])
    model_params = model.parameters()
    for param in model_params:
        print(type(param.data), param.size())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    while True:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1
            batch_size = len(src_sents)

            if train_iter % 5 == 0:
                print("#", end="", flush=True)

            # start training routine
            optimizer.zero_grad()
            loss_v = model(src_sents, tgt_sents)
            loss = torch.sum(loss_v)
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), clip_grad)
            optimizer.step()

            report_loss += loss
            cum_loss += loss.detach()

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cumulative_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cumulative_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cumulative_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), flush=True)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # the following code performs validation on dev set, and controls the learning schedule
            # if the dev score is better than the last check point, then the current model is saved.
            # otherwise, we allow for that performance degeneration for up to `--patience` times;
            # if the dev score does not increase after `--patience` iterations, we reload the previously
            # saved best model (and the state of the optimizer), halve the learning rate and continue
            # training. This repeats for up to `--max-num-trial` times.
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cumulative_examples,
                                                                                         np.exp(cum_loss / cumulative_tgt_words),
                                                                                         cumulative_examples))

                cum_loss = cumulative_examples = cumulative_tgt_words = 0.
                valid_num += 1

                print('begin validation ... size %d %d' % (len(dev_data), len(dev_data_src)))

                # set model to evaluate mode
                model.eval()
                # compute dev. ppl and bleu
                dev_ppl = model.evaluate_ppl(dev_data, batch_size=128)   # dev batch size can be a bit larger
                # set model back to training mode
                model.train()
                '''
                print("dev. ppl %f" % dev_ppl)
                dev_hyps = []
                for dev_src_sent in dev_data_src:
                    print(".", end="", flush=True)
                    dev_hyp_sent = model.beam_search(dev_src_sent)
                    dev_hyps.append(dev_hyp_sent[0])
                dev_bleu = compute_corpus_level_bleu_score(dev_data_tgt, dev_hyps)
                '''
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl))

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path)
                    model.save(model_save_path)
                    torch.save(optimizer, optimizer_save_path)

                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!')
                            exit(0)

                        # load model
                        model = model.load(model_save_path)
                        optimizer = torch.load(optimizer_save_path)

                        # decay learning rate, and restore from previously best checkpoint
                        lr = lr * float(args['--lr-decay'])
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                        print('load previously best model and decay learning rate to %f' % lr)

                        # reset patience
                        patience = 0

                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!')
                    exit(0)


def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) \
        -> List[List[Hypothesis]]:
    hypotheses = []
    for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
        example_hyps = model.beam_search(src_sent, beam_size=beam_size,
                                         max_decoding_time_step=max_decoding_time_step)

        hypotheses.append(example_hyps)

    return hypotheses


def decode(args: Dict[str, str]):
    """
    performs decoding on a test set, and save the best-scoring decoding results. 
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print(f"load model from {args['MODEL_PATH']}")
    model = NMT.load(args['MODEL_PATH'])

    vocab = pickle.load(open('data/vocab.bin', 'rb'))
    model.vocab = vocab
    # set model to evaluate mode
    model.eval()

    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']))

    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print(f'Corpus BLEU: {bleu_score}')

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')


def main():
    args = docopt(__doc__)

    # seed the random number generator (RNG) for numpy and pytorch
    seed = int(args['--seed'])
    np.random.seed(seed * 13 // 7)
    torch.manual_seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError(f'invalid mode')


if __name__ == '__main__':
    main()
