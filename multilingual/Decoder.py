from typing import *
import torch
import torch.nn as nn
import torch.tensor as Tensor
import torch.nn.functional as F


from embed import corpus_to_indices, indices_to_corpus
from FLSTM import Stack_FLSTMCell

class Decoder:
    def __init__(self, device: torch.device, vocab, dropout_rate, embedding_weights: Tensor,
                 Ws: Tensor, Wc: Tensor, Wa: Tensor, embed_size, hidden_size, num_layers, LSTM_cell_weights):
        self.device = device
        self.vocab = vocab
        self.DECODER_PAD_IDX = self.vocab.tgt.word2id['<pad>']
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=2)
        self.criterion = nn.NLLLoss()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.embedding_weights = embedding_weights
        self.Ws = Ws
        self.Wc = Wc
        self.Wa = Wa
        self.tanh = nn.Tanh()
        self.dec_embed_size = embed_size
        self.dec_hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_cell = Stack_FLSTMCell(input_size=embed_size, hidden_size=hidden_size, weights=LSTM_cell_weights,
                                         num_layers=num_layers)

    @staticmethod
    def assert_tenser_size(tensor: Tensor, expected_size: list):
        try:
            assert list(tensor.shape) == expected_size
        except AssertionError:
            print("tensor shape %s doesn't match expected size %s" % (tensor.shape, expected_size))
            raise

    def decode(self, src_encodings: Tensor, decoder_init_state: Tensor, tgt_sents: List[List[str]]) -> Tensor:
        """
        Given source encodings, compute the log-likelihood of predicting the gold-standard target
        sentence tokens

        Args:
            src_encodings: hidden states of tokens in source sentences of shape
            [max_src_len, batch_size, 2 * enc_hidden_size]
            decoder_init_state: decoder GRU/LSTM's initial state
            tgt_sents: list of gold-standard target sentences, wrapped by `<s>` and `</s>`

        Returns:
            scores: could be a variable of shape (batch_size, ) representing the
                log-likelihood of generating the gold-standard target sentence for
                each example in the input batch
                (extra note) we need this to be in the shape of (batch_size, output_vocab_size)
                for beam search
        """
        batch_size = len(tgt_sents)
        # input.shape = [batch_size, 1 (sent_len)]
        input = corpus_to_indices(self.vocab.tgt, ["<s>" for _ in range(batch_size)]).to(self.device)
        # dim = (batch_size, embed_size)
        decoder_input = F.embedding(input, self.embedding_weights)
        self.assert_tenser_size(decoder_input, [batch_size, self.dec_embed_size])
        scores = torch.zeros(batch_size, device=self.device)
        # [num_layers, batch_size, dec_hidden_size]
        h_t = decoder_init_state[0]
        c_t = decoder_init_state[1]
        zero_mask = torch.zeros(batch_size, device=self.device)
        one_mask = torch.ones(batch_size, device=self.device)
        # convert the target sentences to indices, dim = (batch_size, max_tgt_len)
        target_output = corpus_to_indices(self.vocab.tgt, tgt_sents).to(self.device)
        # [batch_size, 2 * enc_hidden_size]
        attn = torch.zeros(h_t.shape[1:], device=self.device)
        # skip the '<s>' in the tgt_sents since the output starts from the word after '<s>'
        for i in range(1, target_output.shape[1]):
            decoder_input = self.dropout(decoder_input)
            h_t, c_t, softmax_output, attn = self.decoder_step(src_encodings, decoder_input, h_t, c_t, attn)
            # dim = (batch_size)
            target_word_indices = target_output[:, i].reshape(batch_size)
            score_delta = self.criterion(softmax_output, target_word_indices)
            # mask '<pad>' with 0
            pad_mask = torch.where((target_word_indices == self.DECODER_PAD_IDX), zero_mask, one_mask)
            masked_score_delta = score_delta * pad_mask
            # update scores
            scores = scores + masked_score_delta
            # get the input for the next layer from the embed of the target words
            decoder_input = F.embedding(target_word_indices, self.embedding_weights)
            self.assert_tenser_size(decoder_input, [batch_size, self.dec_embed_size])
        return scores

    def decoder_step(self, src_encodings: Tensor, decoder_input: Tensor, h_t: Tensor, c_t: Tensor, attn: Tensor):
        """
        Perform one decoder step

        :param src_encodings:  the output features (h_t) from the last layer in source sentences of shape
            [max_src_len, batch_size, 2 * enc_hidden_size]
        :param decoder_input: (batch_size, embed_size)
        :param h_t: [num_layers, batch_size, dec_hidden_size]
        :param c_t: [num_layers, batch_size, dec_hidden_size]
        :param attn: [batch_size, 2 * enc_hidden_size]
        :return: new h_t, c_t, softmax_output with dim (batch_size, vocab_size), attn (batch_size, 2 * enc_hidden_size)
        """
        # dim = (batch_size,  2 * enc_hidden_size + dec_embed_size)
        cat_input = torch.cat((attn, decoder_input), 1)
        h_t, c_t = self.lstm_cell(cat_input, h_t, c_t)
        # dim = (batch_size, dec_hidden_size)
        attn_h_t = self.global_attention(src_encodings, h_t)
        # dim = (batch_size, vocab_size)
        vocab_size_output = F.linear(attn_h_t, self.Ws)
        # dim = (batch_size, vocab_size)
        softmax_output = self.log_softmax(vocab_size_output).squeeze()
        return h_t, c_t, softmax_output, attn_h_t

    def global_attention(self, h_s: Tensor, h_t: list[Tensor]):
        """
        Calculate global attention

        :param h_s: source top hidden state of size [max_src_len, batch_size, 2 * enc_hidden_size]
        :param h_t: a list of decoder hidden state of shape [batch_size, dec_hidden_size]
        :return: an attention vector (batch_size, dec_hidden_size)
        """
        # h_t_top.shape = [batch_size, dec_hidden_size]
        h_t_top = h_t[-1]
        # dim = (batch_size, max_src_len, 2 * enc_hidden_size)
        h_s_ = h_s.transpose(0, 1)
        # dim = (batch_size, 1, max_src_len)
        score = self.general_score(h_s_, h_t_top)
        # dim = (batch_size, 1, max_src_len)
        a_t = self.softmax(score)
        # a_t = self.dropout(a_t)
        # dim = (batch_size, 2 * enc_hidden_size)
        c_t = torch.bmm(a_t, h_s_).squeeze()
        # dim = (batch_size, 2 * enc_hidden_size + dec_hidden_size)
        cat_c_h = torch.cat((c_t, h_t_top), 1)
        return self.tanh(F.linear(cat_c_h, self.Wc))

    def general_score(self, h_s_: Tensor, h_t_top: Tensor):
        """
        Calculate general attention score

        :param h_s_: transposed source top hidden state of size [batch_size, max_src_len, 2 * enc_hidden_size]
        :param h_t_top: decoder hidden state of size [batch_size, dec_hidden_size]
        :return: a score of size (batch_size, 1, max_src_len)
        """
        # dim = (batch_size, max_src_len, 2 * enc_hidden_size)
        W_a_h_s = F.linear(h_s_, self.Wa)
        # dim = (batch_size, 2 * enc_hidden_size, max_src_len)
        W_a_h_s = W_a_h_s.transpose(1, 2)
        return torch.bmm(h_t_top.unsqueeze(1), W_a_h_s)
