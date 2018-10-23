from typing import *
import torch
import torch.nn as nn
import torch.tensor as Tensor
import torch.nn.functional as F

from FLSTM import Stack_FLSTMCell

from config import device
from vocab import Vocab


class Decoder:
    def __init__(self, batch_size, embed_size, hidden_size, num_layers,
                 embedding: nn.Embedding, lstm_weights: List[List[Tensor]], attn_weights: List[List[Tensor]],
                 dropout_rate=0):
        self.embedding = embedding
        self.dec_embed_size = embed_size
        self.dec_hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm_cell = Stack_FLSTMCell(input_size=hidden_size + embed_size, hidden_size=hidden_size,
                                         weights=lstm_weights, num_layers=num_layers)
        self.Wa, self.Wc, self.Ws = attn_weights[0]
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=2)
        self.criterion = nn.NLLLoss()
        self.tanh = nn.Tanh()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)
        sos_batch = torch.tensor([Vocab.SOS_ID for _ in range(batch_size)], dtype=torch.long).to(device)
        self.init_input = embedding(sos_batch)

    @staticmethod
    def init_decoder_step_input(decoder_init_state: Tuple[Tensor, Tensor]) \
            -> Tuple[Tensor, Tensor, Tensor]:
        """
        Initial input to decoder step

        :param d: device type
        :param decoder_init_state: decoder GRU/LSTM's initial state
        :return: h_0, c_0 of shape [num_layers, self.batch_size, dec_hidden_size],
        attn of shape [batch_size, num_direction * enc_hidden_size]
        """
        # [num_layers, self.batch_size, dec_hidden_size]
        h_0 = decoder_init_state[0]
        c_0 = decoder_init_state[1]
        # [batch_size, num_direction * enc_hidden_size]
        attn = torch.zeros(h_0.shape[1:], device=device)
        return h_0, c_0, attn

    def __call__(self, src_encodings: Tensor, decoder_init_state: Tensor, tgt_sent_idx: Tensor) -> Tensor:
        """
        Given source encodings, compute the log-likelihood of predicting the gold-standard target
        sentence tokens

        Args:
            src_encodings: hidden states of tokens in source sentences of shape
            [src_len, batch_size, num_direction * enc_hidden_size]
            decoder_init_state: decoder GRU/LSTM's initial state
            tgt_sent_idx: indices of gold-standard target sentences with dim [batch_size, sent_len]

        Returns:
            scores: could be a variable of shape [batch_size, ] representing the
                log-likelihood of generating the gold-standard target sentence for
                each example in the input batch
                (extra note) we need this to be in the shape of (batch_size, output_vocab_size)
                for beam search
        """
        # dim = (batch_size, embed_size)
        decoder_input = self.init_input
        scores = torch.zeros(self.batch_size, device=device)
        zero_mask = torch.zeros(self.batch_size, device=device)
        one_mask = torch.ones(self.batch_size, device=device)
        h_t, c_t, attn = self.init_decoder_step_input(decoder_init_state)
        # dim = (batch_size, sent_len, embed_size)
        tgt_sent_embed = self.embedding(tgt_sent_idx)
        top_subwords = [[] for _ in range(self.batch_size)]
        # skip the '<s>' in the tgt_sents since the output starts from the word after '<s>'
        for i in range(1, tgt_sent_idx.shape[1]):
            decoder_input = self.dropout(decoder_input)
            h_t, c_t, softmax_output, attn = self.decoder_step(src_encodings, decoder_input, h_t, c_t, attn)
            _, top_id = torch.topk(softmax_output, 1, dim=1)
            for j in range(self.batch_size):
                top_subwords[j].append(int(top_id[j]))
            # dim = (batch_size)
            target_word_indices = tgt_sent_idx[:, i].reshape(self.batch_size)
            score_delta = self.criterion(softmax_output, target_word_indices)
            # mask the <pad> with zero
            pad_mask = torch.where((target_word_indices == Vocab.PAD_ID), zero_mask, one_mask)
            masked_score_delta = score_delta * pad_mask
            # update scores
            scores = scores + masked_score_delta
            # dim = (batch_size, embed_size)
            decoder_input = tgt_sent_embed[:, i, :]
        return scores, top_subwords

    def decoder_step(self, src_encodings: Tensor, decoder_input: Tensor, h_t: Tensor, c_t: Tensor, attn: Tensor)\
            -> (Tensor, Tensor, Tensor, Tensor):
        """
        Perform one decoder step

        :param src_encodings:  the output features (h_t) from the last layer in source sentences of shape
            [batch_size, src_len, num_direction * enc_hidden_size]
        :param decoder_input: (batch_size, embed_size)
        :param h_t: [num_layers, batch_size, dec_hidden_size]
        :param c_t: [num_layers, batch_size, dec_hidden_size]
        :param attn: [batch_size, num_direction * enc_hidden_size]
        :return: new h_t, c_t, softmax_output with dim (batch_size, vocab_size), attn (batch_size, num_direction * enc_hidden_size)
        """
        h_t, c_t = self.lstm_cell(
            torch.cat((attn, decoder_input), 1),  # [batch_size,  num_direction * enc_hidden_size + dec_embed_size]
            h_t, c_t)
        # attn_h_t.shape = [batch_size, dec_hidden_size]
        attn_h_t = self.global_attention(src_encodings, h_t[-1])
        # softmax_output.shape = [batch_size, vocab_size]
        softmax_output = self.log_softmax(
            F.linear(attn_h_t, self.Ws))  # [batch_size, vocab_size]
        return h_t, c_t, softmax_output, attn_h_t

    def global_attention(self, h_s: Tensor, h_t_top: Tensor) -> Tensor:
        """
        Calculate global attention

        :param h_s: source top hidden state of size [batch_size, src_len, num_direction * enc_hidden_size]
        :param h_t_top: decoder hidden state of size [batch_size, dec_hidden_size]
        :return: an attention vector (batch_size, dec_hidden_size)
        """
        # dim = (batch_size, num_direction * enc_hidden_size)
        c_t = torch.bmm(self.softmax(self.general_score(h_s, h_t_top)), h_s)[:, 0, :]
        # dim = (batch_size, num_direction * enc_hidden_size + dec_hidden_size)
        cat_c_h = torch.cat((c_t, h_t_top), 1)
        return self.tanh(F.linear(cat_c_h, self.Wc))

    def general_score(self, h_s: Tensor, h_t_top: Tensor) -> Tensor:
        """
        Calculate general attention score

        :param h_s: source top hidden state of size [batch_size, src_len, num_direction * enc_hidden_size]
        :param h_t_top: decoder hidden state of size [batch_size, dec_hidden_size]
        :return: a score of size (batch_size, 1, src_len)
        """
        return torch.bmm(h_t_top.unsqueeze(1),
                         # [batch_size, num_direction * enc_hidden_size = dec_hidden_size, src_len]
                         F.linear(h_s, self.Wa).transpose(1, 2))
