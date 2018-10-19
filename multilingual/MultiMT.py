from typing import *

import torch
import torch.nn as nn
import torch.tensor as Tensor

from CPG import CPG
from Encoder import Encoder
from Decoder import Decoder


class NMT(nn.Module):
    def __init__(self, size_dict: Dict[str, int]):
        super(NMT, self).__init__()

        # init size constants
        self.embed_size = size_dict['embed_size']
        self.hidden_size = size_dict['hidden_size']
        self.vocab_size = size_dict['vocab_size']
        self.num_layers = size_dict['num_layers']
        self.batch_size = size_dict['batch_size']
        self.NUM_DIR = 2
        # init encoder param shapes
        self.enc_in_lstm_shapes = NMT.get_shapes_flstm(self.embed_size, self.hidden_size, self.num_layers)
        self.enc_rev_lstm_shapes = NMT.get_shapes_flstm(self.embed_size, self.hidden_size, self.num_layers)
        self.enc_shapes = self.enc_in_lstm_shapes + self.enc_rev_lstm_shapes
        self.enc_shapes_len = len(self.enc_shapes)
        # init decoder param shapes
        self.decoder_hidden_size = self.NUM_DIR * self.hidden_size
        self.dec_lstm_shapes = NMT.get_shapes_flstm(self.embed_size, self.decoder_hidden_size, self.num_layers)
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
        self.cpg = CPG(self.param_shapes, size_dict)

    def forward(self, src_sent_idx: Tensor, tgt_sent_idx: Tensor, src_lang: int, tgt_lang: int) -> Tensor:
        """
        Takes in a batch of paired src and tgt sentences with lang tags, return the loss

        :param src_sent_idx: dim = (batch_size, sent_length)
        :param tgt_sent_idx: dim = (batch_size, sent_length)
        :param src_lang: src language index
        :param tgt_lang: target language index
        :return: scores with dim = (batch_size)
        """
        # create a list of language indices corresponding each param group
        langs = [src_lang for _ in range(self.enc_shapes_len)] + [tgt_lang for _ in range(self.dec_shapes_len)]
        grouped_params = self.cpg.get_params(langs)
        # encode
        enc_weights = grouped_params[:self.enc_shapes_len]
        encoder = Encoder(self.batch_size, self.embed_size, self.hidden_size, self.cpg.get_embedding(src_lang),
                          enc_weights)
        src_encodings, decoder_init_state = encoder(src_sent_idx)
        # decode
        dec_lstm_weights = grouped_params[self.enc_shapes_len:self.enc_shapes_len + self.dec_lstm_shapes_len]
        attn_weights = grouped_params[self.enc_shapes_len + self.dec_lstm_shapes_len:]
        decoder = Decoder(self.batch_size, self.embed_size, self.hidden_size, self.cpg.get_embedding(tgt_lang),
                          dec_lstm_weights, attn_weights)
        return decoder(src_encodings, decoder_init_state, tgt_sent_idx, init_input)

    def infer(self, src_sent: Tensor, src_lang: int, tgt_lang: int) -> Tensor:
        """
        Takes in ONE src sentence with language tag, return the corresponding translation (word indices)
        :param src_sent: dim = (1, sent_length)
        :param src_lang: source language index
        :param tgt_lang: target language index
        :return: di = ( 1 * ? ) where ? represents the translated sentence length
        """

        pass

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
