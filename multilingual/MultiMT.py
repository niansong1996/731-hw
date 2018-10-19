from typing import *

import numpy as np
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

        # init param shapes
        self.enc_in_lstm = NMT.get_shapes_flstm(self.embed_size, self.hidden_size, self.num_layers)
        self.enc_rev_lstm = NMT.get_shapes_flstm(self.embed_size, self.hidden_size, self.num_layers)
        self.dec_lstm = NMT.get_shapes_flstm(self.embed_size, 2*self.hidden_size, self.num_layers)
        self.dec_attn = [[(1,1), (1,1)]] # TODO: set this to attn dims

        self.param_shapes = self.enc_in_lstm + self.enc_rev_lstm + self.dec_lstm + self.dec_attn

        # init CPG
        self.cpg = CPG(self.param_shapes, size_dict)


    def forward(self, src_sent: Tensor, tgt_sent: Tensor, src_lang: int, tgt_lang: int) -> Tensor:
        """
        take in a batch of paired src and tgt sentences with lang tags, return the loss

        :param src_sent: dim = (batch_size, sent_length)
        :param tgt_sent: dim = (batch_size, sent_length)
        :param src_lang: src language index
        :param tgt_lang: target language index
        :return: a scalar in tensor form
        """

        pass


    def infer(self, src_sent: Tensor, src_lang: int, tgt_lang: int) -> Tensor:
        """
        take in ONE src sentence with language tag, return corresponding translation (word indices)
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




