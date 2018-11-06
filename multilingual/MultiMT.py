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
from utils import batch_iter, PairedData, sents_to_tensor, assert_tensor_size
from vocab import Vocab

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class MultiNMT(nn.Module):
    def __init__(self, args: Dict[str, str]):
        super(MultiNMT, self).__init__()
        # init size constants
        self.embed_size = int(args['--embed-size'])
        self.hidden_size = int(args['--hidden-size'])
        self.vocab_size = int(args['--vocab-size'])
        self.num_layers = int(args['--num-layers'])
        self.dropout_rate = float(args['--dropout'])
        self.denoising_rate = float(args['--denoising'])
        self.NUM_DIR = 2
        # init encoder param shapes
        self.enc_in_lstm_shapes = MultiNMT.get_shapes_flstm(self.embed_size, self.hidden_size, self.num_layers)
        self.enc_rev_lstm_shapes = MultiNMT.get_shapes_flstm(self.embed_size, self.hidden_size, self.num_layers)
        self.enc_shapes = self.enc_in_lstm_shapes + self.enc_rev_lstm_shapes
        self.enc_shapes_len = len(self.enc_shapes)
        # init decoder param shapes
        self.decoder_hidden_size = self.NUM_DIR * self.hidden_size
        self.decoder_input_size = self.decoder_hidden_size + self.embed_size
        self.dec_lstm_shapes = MultiNMT.get_shapes_flstm(self.decoder_input_size, self.decoder_hidden_size,
                                                         self.num_layers)
        self.dec_lstm_shapes_len = len(self.dec_lstm_shapes)
        decoder_W_a_shape = (self.decoder_hidden_size, self.NUM_DIR * self.hidden_size)
        decoder_W_c_shape = (self.decoder_hidden_size, self.NUM_DIR * self.hidden_size + self.decoder_hidden_size)
        decoder_W_s_shape = (self.vocab_size, self.decoder_hidden_size)
        self.dec_attn_shapes = [[decoder_W_a_shape, decoder_W_c_shape, decoder_W_s_shape]]
        self.dec_shapes = self.dec_lstm_shapes + self.dec_attn_shapes
        self.dec_shapes_len = len(self.dec_shapes)
        # combine enc and dec param shapes
        self.param_shapes = self.enc_shapes + self.dec_shapes
        # init CPG
        self.cpg = CPG(self.param_shapes, args, self.enc_shapes_len)

    def forward(self, src_lang: int, tgt_lang: int, src_sents: List[List[int]], tgt_sents: List[List[int]]) \
            -> Tensor:
        """
        Takes in a batch of paired src and tgt sentences with lang tags, return the loss

        :param src_lang: source language index
        :param tgt_lang: target language index
        :param src_sents: batch_size of sentences
        :param tgt_sents: batch_size of sentences
        :return: scores with shape = [batch_size]
        """
        # [batch_size, sent_len]
        src_sents_tensor = sents_to_tensor(src_sents, device)
        # [batch_size, sent_len]
        tgt_sents_tensor = sents_to_tensor(tgt_sents, device)
        assert (src_sents_tensor.shape[0] == tgt_sents_tensor.shape[0])
        batch_size = src_sents_tensor.shape[0]
        grouped_params = self.get_grouped_params(src_lang, tgt_lang)
        enc_lstm_weights = grouped_params[:self.enc_shapes_len]
        dec_lstm_weights = grouped_params[self.enc_shapes_len:self.enc_shapes_len + self.dec_lstm_shapes_len]
        attn_weights = grouped_params[self.enc_shapes_len + self.dec_lstm_shapes_len:]
        src_embedding = self.cpg.get_embedding(src_lang)
        tgt_embedding = self.cpg.get_embedding(tgt_lang)
        # encode
        src_encodings, decoder_init_state = self.encode(batch_size, src_sents_tensor, src_lang, src_embedding, enc_lstm_weights)
        # decode
        dec_lstm_weights = grouped_params[self.enc_shapes_len:self.enc_shapes_len + self.dec_lstm_shapes_len]
        attn_weights = grouped_params[self.enc_shapes_len + self.dec_lstm_shapes_len:]
        decoder = self.get_decoder(tgt_lang, batch_size)
        return decoder(src_encodings, decoder_init_state, tgt_sents_tensor, tgt_embedding, dec_lstm_weights, attn_weights)

    def get_grouped_params(self, src_lang: int, tgt_lang: int) -> List[List[Tensor]]:
        # create a list of language indices corresponding each param group
        langs = [src_lang for _ in range(self.enc_shapes_len)] + [src_lang for _ in range(self.dec_shapes_len)]
        return self.cpg.get_params(langs)

    def encode(self, batch_size: int, src_sent_idx: Tensor, src_lang: int, src_embedding, enc_lstm_weights: List[List[Tensor]]) \
            -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """

        :param src_sent_idx: source sentence word indices dim = (batch_size, sent_len)
        :param src_lang: source language index
        :param grouped_params: a list of groups of parameters in tensor form
        :return: outputs: shape = [sent_length, batch_size, num_direction * hidden_size]
            h_t, c_t: shape = [num_layers, batch_size, num_direction * hidden_size]
        """
        encoder = Encoder(batch_size, self.embed_size, self.hidden_size, 
                           self.training, self.dropout_rate, num_layer=self.num_layers)
        return encoder(src_sent_idx, src_embedding, enc_lstm_weights)

    def get_decoder(self, tgt_lang: int, batch_size: int) -> Decoder:
        return Decoder(batch_size, self.embed_size, self.decoder_hidden_size, self.num_layers,
                       self.cpg.get_embedding(tgt_lang), training=self.training, dropout_rate=self.dropout_rate)

    def beam_search(self, src_sent: List[int], src_lang: int, tgt_lang: int, beam_size: int=5,
                    max_decoding_time_step: int=70) -> Tensor:
        """
        Takes in ONE src sentence with language tag, return the corresponding translation (word indices)
        :param src_sent: batch_size of sentences
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
            enc_lstm_weights = grouped_params[:self.enc_shapes_len]
            dec_lstm_weights = grouped_params[self.enc_shapes_len:self.enc_shapes_len + self.dec_lstm_shapes_len]
            attn_weights = grouped_params[self.enc_shapes_len + self.dec_lstm_shapes_len:]
            src_embedding = self.cpg.get_embedding(src_lang)
            tgt_embedding = self.cpg.get_embedding(tgt_lang)
            # [batch_size, sent_len]
            src_sents_tensor = sents_to_tensor([src_sent], device)
            # src_encodings.shape = [sent_length, 1, embed_size]
            src_encodings, decoder_init_state = self.encode(1, src_sents_tensor, src_lang, src_embedding, enc_lstm_weights)
            h_t_0, c_t_0, attn = Decoder.init_decoder_step_input(decoder_init_state)
            # candidates for best hypotheses
            hypotheses_cand = [(Hypothesis([Vocab.SOS_ID], 0), h_t_0, c_t_0, attn)]
            decoder = self.get_decoder(tgt_lang, 1)
            for i in range(max_decoding_time_step):
                new_hypotheses_cand = []
                for (sent, log_likelihood), h_t, c_t, attn in hypotheses_cand:
                    input_word_idx = sent[-1]
                    # skip ended sentence
                    if input_word_idx == Vocab.EOS_ID:
                        # directly add ended sentence to new candidates
                        new_hypotheses_cand.append((Hypothesis(sent, log_likelihood), h_t, c_t, attn))
                        continue
                    # dim = (1 (single_word), embed_size)
                    decoder_input = tgt_embedding(torch.tensor([input_word_idx]).to(device))
                    assert_tensor_size(decoder_input, [1, self.embed_size])
                    # softmax_output.shape = [1, vocab_size]
                    h_t, c_t, softmax_output, attn = decoder.decoder_step(src_encodings, decoder_input, h_t, c_t, attn, dec_lstm_weights, attn_weights)
                    # dim = (1, beam_size)
                    _, top_i = torch.topk(softmax_output, beam_size, dim=1)
                    for word_idx_tensor in top_i[0]:
                        word_idx = word_idx_tensor.item()
                        new_hyp = Hypothesis(sent + [word_idx], log_likelihood + float(softmax_output[0][word_idx]))
                        new_hypotheses_cand.append((new_hyp, h_t, c_t, attn))
                # combine ended sentences with new candidates to form new hypotheses
                hypotheses_cand = sorted(new_hypotheses_cand, key=lambda x: x[0].score, reverse=True)[:beam_size]
                # break if all sentences have ended
                if all(c[0].value[-1] == Vocab.EOS_ID for c in hypotheses_cand):
                    break
            return [c[0] for c in hypotheses_cand]

    def save(self, path: str):
        torch.save(self, path)

    @staticmethod
    def load(model_path: str):
        loaded_model = torch.load(model_path)
        #for param in loaded_model.cpg.word_embeddings.parameters():
        #    param.requires_grad = False
        return loaded_model

    @staticmethod
    def get_shapes_flstm(input_size, hidden_size, num_layers):
        params_in_lstm = []

        for i in range(num_layers):
            params_in_group = []
            input_size = input_size if i == 0 else hidden_size
            params_in_group.append((4 * hidden_size, input_size))
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
        decoded_sents = []
        reference_sents = []
        with torch.no_grad():
            for src_lang, tgt_lang, src_sents, tgt_sents in batch_iter(dev_data, batch_size, shuffle=False):
                loss, sents = self(src_lang, tgt_lang, src_sents, tgt_sents)
                loss = loss.sum()
                # calculate the ppl.
                cum_loss += loss
                tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting the leading `<s>`
                cum_tgt_words += tgt_word_num_to_predict

                # formulate the decoded sent
                decoded_sents += [sent.numpy() for sent in sents]
                reference_sents += tgt_sents

            ppl = np.exp(cum_loss / cum_tgt_words)

            for i in range(len(decoded_sents)):
                eos = np.argmax(decoded_sents[i]==Vocab.EOS_ID)
                if not eos == 0:
                    decoded_sents[i] = decoded_sents[i][:eos+1]
                decoded_sents[i] = list(map(int, decoded_sents[i].tolist()))

            return ppl, (reference_sents, decoded_sents)
