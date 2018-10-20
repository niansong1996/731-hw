from collections import namedtuple
from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.tensor as Tensor

from CPG import CPG
from Decoder import Decoder
from Encoder import Encoder
from config import device
from utils import batch_iter, PairedData
from vocab import VocabEntry

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class MultiNMT(nn.Module):
    def __init__(self, args: Dict[str, str]):
        super(MultiNMT, self).__init__()
        # init size constants
        self.embed_size = int(args['--embed-size']),
        self.hidden_size = int(args['--hidden-size']),
        self.vocab_size = int(args['--vocab_size']),
        self.num_layers = int(args['--num_layers']),
        self.train_batch_size = int(args['--batch-size'])
        self.dropout_rate = float(args['--dropout'])
        self.NUM_DIR = 2
        # init encoder param shapes
        self.enc_in_lstm_shapes = MultiNMT.get_shapes_flstm(self.embed_size, self.hidden_size, self.num_layers)
        self.enc_rev_lstm_shapes = MultiNMT.get_shapes_flstm(self.embed_size, self.hidden_size, self.num_layers)
        self.enc_shapes = self.enc_in_lstm_shapes + self.enc_rev_lstm_shapes
        self.enc_shapes_len = len(self.enc_shapes)
        # init decoder param shapes
        self.decoder_hidden_size = self.NUM_DIR * self.hidden_size
        self.dec_lstm_shapes = MultiNMT.get_shapes_flstm(self.embed_size, self.decoder_hidden_size, self.num_layers)
        self.dec_lstm_shapes_len = len(self.dec_lstm_shapes)
        decoder_W_a_shape = (self.NUM_DIR * self.hidden_size, self.decoder_hidden_size)
        decoder_W_c_shape = (self.NUM_DIR * self.hidden_size + self.decoder_hidden_size, self.decoder_hidden_size)
        decoder_W_s_shape = (self.decoder_hidden_size, self.vocab_size)
        self.dec_attn_shapes = [[decoder_W_a_shape, decoder_W_c_shape, decoder_W_s_shape]]
        self.dec_shapes = self.dec_lstm_shapes + self.dec_attn_shapes
        self.dec_shapes_len = len(self.dec_shapes)
        # combine enc and dec param shapes
        self.param_shapes = self.enc_shapes + self.dec_shapes
        # init CPG
        self.cpg = CPG(self.param_shapes, args)

    def forward(self, src_lang: int, tgt_lang: int, src_sent_idx: Tensor, tgt_sent_idx: Tensor) -> Tensor:
        """
        Takes in a batch of paired src and tgt sentences with lang tags, return the loss

        :param src_lang: source language index
        :param tgt_lang: target language index
        :param src_sent_idx: [batch_size, sent_length]
        :param tgt_sent_idx: [batch_size, sent_length]
        :return: scores with shape = [batch_size]
        """
        grouped_params = self.get_grouped_params(src_lang, tgt_lang)
        # encode
        src_encodings, decoder_init_state = self.encode(src_sent_idx, src_lang, grouped_params)
        # decode
        decoder = self.get_decoder(tgt_lang, grouped_params)
        return decoder(src_encodings, decoder_init_state, tgt_sent_idx)

    def get_grouped_params(self, src_lang: int, tgt_lang: int) -> List[List[Tensor]]:
        # create a list of language indices corresponding each param group
        langs = [src_lang for _ in range(self.enc_shapes_len)] + [tgt_lang for _ in range(self.dec_shapes_len)]
        return self.cpg.get_params(langs)

    def encode(self, src_sent_idx: Tensor, src_lang: int, grouped_params: List[List[Tensor]]) \
            -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """

        :param src_sent_idx: source sentence word indices dim = (batch_size, sent_len)
        :param src_lang: source language index
        :param grouped_params: a list of groups of parameters in tensor form
        :return: outputs: shape = [sent_length, batch_size, num_direction * hidden_size]
            h_t, c_t: shape = [num_layers, batch_size, num_direction * hidden_size]
        """
        enc_weights = grouped_params[:self.enc_shapes_len]
        encoder = Encoder(self.batch_size, self.embed_size, self.hidden_size, self.cpg.get_embedding(src_lang),
                          enc_weights)
        return encoder(src_sent_idx)

    def get_decoder(self, tgt_lang: int, grouped_params: List[List[Tensor]]) -> Decoder:
        dec_lstm_weights = grouped_params[self.enc_shapes_len:self.enc_shapes_len + self.dec_lstm_shapes_len]
        attn_weights = grouped_params[self.enc_shapes_len + self.dec_lstm_shapes_len:]
        return Decoder(self.batch_size, self.embed_size, self.decoder_hidden_size, self.num_layers,
                       self.cpg.get_embedding(tgt_lang), dec_lstm_weights, attn_weights)

    def beam_search(self, src_sent_idx: Tensor, src_lang: int, tgt_lang: int, beam_size: int=5,
                    max_decoding_time_step: int=70) -> Tensor:
        """
        Takes in ONE src sentence with language tag, return the corresponding translation (word indices)
        :param src_sent_idx: dim = (1, sent_length)
        :param src_lang: source language index
        :param tgt_lang: target language index
        :param beam_size: beam size
        :param max_decoding_time_step: maximum number of time steps to unroll the decoding RNN
        :return: hypotheses: a list of hypothesis of beam_size, each hypothesis has two fields:
                value: List[int]: the decoded target sentence, represented as a list of word index
                score: float: the log-likelihood of the target sentence
        """
        with torch.no_grad():
            grouped_params = self.get_grouped_params(src_lang, tgt_lang)
            # src_encodings.shape = [sent_length, 1, embed_size]
            src_encodings, decoder_init_state = self.encode(src_sent_idx, src_lang, grouped_params)
            h_t_0, c_t_0, attn = Decoder.init_decoder_step_input(decoder_init_state)
            # candidates for best hypotheses
            hypotheses_cand = [(Hypothesis([VocabEntry.SOS_ID], 0), h_t_0, c_t_0, attn)]
            decoder = self.get_decoder(tgt_lang, grouped_params)
            for i in range(max_decoding_time_step):
                new_hypotheses_cand = []
                for (sent, log_likelihood), h_t, c_t, attn in hypotheses_cand:
                    input_word_idx = sent[-1]
                    # skip ended sentence
                    if input_word_idx == VocabEntry.EOS_ID:
                        # directly add ended sentence to new candidates
                        new_hypotheses_cand.append((Hypothesis(sent, log_likelihood), h_t, c_t, attn))
                        continue
                    # dim = (1 (single_word), embed_size)
                    decoder_input = self.decoder_embed(torch.tensor([input_word_idx]).to(device))
                    # softmax_output.shape = [vocab_size]
                    h_t, c_t, softmax_output, attn = decoder.decoder_step(src_encodings, decoder_input, h_t, c_t, attn)
                    # dim = (1, beam_size)
                    _, top_i = torch.topk(softmax_output.unsqueeze(0), beam_size, dim=1)
                    for word_idx_tensor in top_i[0]:
                        word_idx = word_idx_tensor.item()
                        new_hyp = Hypothesis(sent + [word_idx], log_likelihood + softmax_output[word_idx])
                        new_hypotheses_cand.append((new_hyp, h_t, c_t, attn))
                # combine ended sentences with new candidates to form new hypotheses
                hypotheses_cand = sorted(new_hypotheses_cand, key=lambda x: x[0].score, reverse=True)[:beam_size]
                # break if all sentences have ended
                if all(c[0].value[-1] == VocabEntry.EOS_ID for c in hypotheses_cand):
                    break
            return [c[0] for c in hypotheses_cand]

    def save(self, path: str):
        torch.save(self, path)

    @staticmethod
    def load(model_path: str):
        return torch.load(model_path)

    @staticmethod
    def get_shapes_flstm(embed_size, hidden_size, num_layers):
        params_in_lstm = []

        for _ in range(num_layers):
            params_in_group = []

            params_in_group.append((4 * hidden_size, embed_size))
            params_in_group.append((4 * hidden_size, hidden_size))
            params_in_group.append((4 * hidden_size, 1))
            params_in_group.append((4 * hidden_size, 1))

            params_in_lstm.append(params_in_group)

        return params_in_lstm

    def evaluate_ppl(self, dev_data: List[PairedData], batch_size: int=32):
        """
        Evaluate perplexity on dev sentences

        Args:
            dev_data: a list of dev sentences
            batch_size: batch size

        Returns:
            ppl: the perplexity on dev sentences
        """
        cum_loss = 0.
        cum_tgt_words = 0.
        with torch.no_grad():
            for src_lang, tgt_lang, src_sents, tgt_sents in batch_iter(dev_data, batch_size):
                loss = self(src_lang, tgt_lang, src_sents, tgt_sents).sum()
                cum_loss += loss
                tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting the leading `<s>`
                cum_tgt_words += tgt_word_num_to_predict

            ppl = np.exp(cum_loss / cum_tgt_words)

            return ppl
