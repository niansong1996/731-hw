# coding=utf-8

"""
A very basic implementation of neural machine translation

Usage:
    nmt.py train --vocab-size=<int> [options]
    nmt.py decode [options] MODEL_PATH SRC_LANG TGT_LANG OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --tune                                  load the model and begin tuning
    --pretrain-model=<file>                 the directory of the pretrained model
    --langs=<src-tgt,...>                   comma separated language pairs <src-tgt>
    --cuda                                  use GPU
    --vocab-size=<int>                      vocab size [default: 20000]
    --low-rank=<int>                        low rank size [default: 4]
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --lang-embed-size=<int>                 language embedding size [default: 8]
    --embed-size=<int>                      word embedding size [default: 256]
    --num-layers=<int>                      number of layers [default: 2]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path
    --save-opt=<file>                       optimizer state save path
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --autoencode-epoch=<int>                using autoencode in the first few epochs [default: 5]
    --dropout=<float>                       dropout [default: 0]
    --denoising=<float>                     percentage of noise in denoising autoencode training [default: 0.2]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
"""

import math
import sys
import time
from typing import *
import pickle
import os.path

import numpy as np
import torch
from docopt import docopt
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

from MultiMT import Hypothesis, MultiNMT
from config import device, LANG_INDICES, LANG_NAMES
from subword import get_corpus_pairs, get_corpus_ids, decode_corpus_ids
from utils import batch_iter, PairedData, LangPair


def get_data_pairs(langs: List[List[str]], data_type: str):
    data = []
    for src_name, tgt_name in langs:
        src = LANG_INDICES[src_name]
        tgt = LANG_INDICES[tgt_name]
        data_pair = get_corpus_pairs(src, tgt, data_type)
        data.append(PairedData(data_pair, LangPair(src, tgt)))
        print('Done loading %s data for %s-%s parallel translation' \
              % (data_type, src_name, tgt_name))
    return data


def train(args: Dict[str, str]):
    lang_pairs = args['--langs']

    # identify translation and autoencode tasks
    langs = [p.split('-') for p in lang_pairs.split(',')]

    # load data from prev dump
    train_file = 'data/train.%s.dump' % lang_pairs
    dev_file = 'data/dev.%s.dump' % lang_pairs
    if os.path.isfile(train_file) and os.path.isfile(dev_file):
        train_data = pickle.load(open(train_file, 'rb'))
        dev_datasets = pickle.load(open(dev_file, 'rb'))
        print("Done reading from dump")
    else:
        train_data = get_data_pairs(langs, 'train')
        dev_datasets = [get_data_pairs([lang], 'dev') for lang in langs]
        pickle.dump(train_data, open(train_file, 'wb'))
        pickle.dump(dev_datasets, open(dev_file, 'wb'))

    train_batch_size = int(args['--batch-size'])
    autoencode_epoch = int(args['--autoencode-epoch'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']
    optimizer_save_path = args['--save-opt']

    # initialize the model
    print('Model initializing...')
    model = MultiNMT(args).to(device)
    if args['--tune']:
        print('tuning mode... load model from %s' % args['--pretrain-model'])
        model = model.load(args['--pretrain-model'])

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cumulative_tgt_words = report_tgt_words = 0
    cumulative_examples = report_examples = epoch = valid_num = 0
    task_hist_valid_scores = [-1e10 for _ in range(len(dev_datasets))]
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    # set the optimizers
    lr = float(args['--lr'])
    model_params = model.named_parameters()
    for param in model_params:
        print(param[0], param[1].size())
        if args['--tune'] and False:
            if param[0] == 'cpg.word_embeddings.0.weight' or param[0] == 'cpg.L.weight':
                param[1].requires_grad = False
                print("freezing %s" % param[0])
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)

    while True:
        epoch += 1
        # remove autoencode training after this point
        if epoch == autoencode_epoch+1:
            train_data = list(filter(lambda x: x.langs.src != x.langs.tgt, train_data))
            print('Stop autoencoding, now training set size -> (%d)' % len(train_data))

        for src_lang, tgt_lang, src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size):
            train_iter += 1
            batch_size = len(src_sents)

            if train_iter % 5 == 0:
                print("#", end="", flush=True)

            # start training routine
            #torch.cuda.empty_cache()
            optimizer.zero_grad()
            loss_v, _ = model(src_lang, tgt_lang, src_sents, tgt_sents)
            loss = torch.sum(loss_v)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            report_loss += float(loss)
            cum_loss += float(loss)
            del loss
            with torch.no_grad():
                tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
                report_tgt_words += tgt_words_num_to_predict
                cumulative_tgt_words += tgt_words_num_to_predict
                report_examples += batch_size
                cumulative_examples += batch_size

                if train_iter % log_every == 0:
                    print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f '
                          'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' %
                          (epoch, train_iter, report_loss / report_examples, math.exp(report_loss / report_tgt_words),
                           cumulative_examples, report_tgt_words / (time.time() - train_time), time.time() - begin_time),
                          flush=True)

                    train_time = time.time()
                    report_loss = report_tgt_words = report_examples = 0.

                # the following code performs validation on dev set, and controls the learning schedule
                # if the dev score is better than the last check point, then the current model is saved.
                # otherwise, we allow for that performance degeneration for up to `--patience` times;
                # if the dev score does not increase after `--patience` iterations, we reload the previously
                # saved best model (and the state of the optimizer), halve the learning rate and continue
                # training. This repeats for up to `--max-num-trial` times.
                if train_iter % valid_niter == 0:
                    print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' %
                          (epoch, train_iter, cum_loss / cumulative_examples, np.exp(cum_loss / cumulative_tgt_words),
                           cumulative_examples))

                    cum_loss = cumulative_examples = cumulative_tgt_words = 0.
                    valid_num += 1

                    print('begin validation ... size %d' % len(dev_datasets))

                    # set model to evaluate mode
                    model.eval()
                    # compute dev. ppl and bleu
                    dev_ppls = []
                    dev_bleus = []
                    for i in range(len(dev_datasets)):
                        dev_data = dev_datasets[i]
                        src_lang_name = LANG_NAMES[dev_data[0].langs.src]

                        pair_name = ("%s-%s" % (src_lang_name, LANG_NAMES[dev_data[0].langs.tgt]))
                        dev_ppl, decodes = model.evaluate_ppl(dev_data, batch_size=128)  # dev batch size can be a bit larger
                        reference_sents = decode_corpus_ids(lang_name=LANG_NAMES[tgt_lang], sents=decodes[0])
                        decoded_sents = decode_corpus_ids(lang_name=LANG_NAMES[tgt_lang], sents=decodes[1])
                        assert len(reference_sents) == len(decoded_sents)
                        dev_bleu = compute_corpus_level_bleu_score(reference_sents, decoded_sents)

                        # only save and evaluate for non-autoencode pairs
                        more_info = ''
                        if not dev_data[0].langs.src == dev_data[0].langs.tgt:
                            dev_ppls.append(float(dev_ppl))
                            dev_bleus.append(dev_bleu)
                            # save best model for specific lang
                            if dev_bleu > task_hist_valid_scores[i]:
                                task_hist_valid_scores[i] = dev_bleu
                                task_model_save_path = model_save_path + ('.%s' % src_lang_name)
                                more_info = '(saved to [%s])' % task_model_save_path
                                model.save(task_model_save_path)

                        print("lang pair %s: dev. ppl %.3f; dev. bleu %.3f %s" % (pair_name, float(dev_ppl), dev_bleu, more_info))

                    # set model back to training mode
                    model.train()

                    dev_bleu = np.mean([float(v) for v in dev_bleus])
                    dev_ppl = np.mean([float(v) for v in dev_ppls])
                    print('validation: iter %d, avg. dev. ppl %f, avg. dev. bleu %f' % (train_iter, dev_ppl, dev_bleu))

                    valid_metric = dev_bleu
                    is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                    hist_valid_scores.append(valid_metric)

                    if is_better:
                        patience = 0
                        print('save currently the best model to [%s]' % model_save_path)
                        model.save(model_save_path)
                        torch.save(optimizer.state_dict(), optimizer_save_path)

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
                            optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
                            optimizer.load_state_dict(torch.load(optimizer_save_path))

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

def compute_corpus_level_bleu_score(references: List[str], hypotheses: List[str]) -> float:
    """
    Given decoding results and reference sentences, compute corpus-level BLEU score

    Args:
        references: a list of gold-standard reference target sentences
        hypotheses: a list of hypotheses, one for each reference

    Returns:
        bleu_score: corpus-level BLEU score
    """
    references = [ref.split(' ') for ref in references]
    hypotheses = [hyp.split(' ') for hyp in hypotheses]

    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp for hyp in hypotheses])

    return bleu_score * 100.0

def beam_search(model: MultiNMT, test_data_src: List[List[int]], src_lang: int, tgt_lang: int, \
                beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:
    hypotheses = []
    for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
        example_hyps = model.beam_search(src_sent, src_lang, tgt_lang, beam_size=beam_size,
                                         max_decoding_time_step=max_decoding_time_step)

        hypotheses.append(example_hyps)

    return hypotheses


def decode(args: Dict[str, str]):
    """
    performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """

    src_lang = args['SRC_LANG']
    tgt_lang = args['TGT_LANG']
    src_lang_idx = LANG_INDICES[src_lang]
    tgt_lang_idx = LANG_INDICES[tgt_lang]

    model_path = args['MODEL_PATH']
    output_file = args['OUTPUT_FILE']

    test_data_src, _ = get_corpus_ids(src_lang_idx, tgt_lang_idx, data_type='test', is_tgt=False, is_train=False)
    #test_data_tgt, _ = get_corpus_ids(src_lang_idx, tgt_lang_idx, data_type='test', is_tgt=True, is_train=False)

    print(f"load model from {model_path}")
    model = MultiNMT.load(model_path)

    # set model to evaluate mode
    model.eval()

    hypotheses = beam_search(model, test_data_src, src_lang_idx, tgt_lang_idx,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']))

    top_hypotheses = [hyps[0].value for hyps in hypotheses]
    translated_text = decode_corpus_ids(lang_name=tgt_lang, sents=top_hypotheses)

    with open(output_file, 'w') as f:
        for sent in translated_text:
            f.write(sent + '\n')


def main():
    args = docopt(__doc__)

    # seed the random number generator (RNG), you may
    # also want to seed the RNG of tensorflow, pytorch, dynet, etc.
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
