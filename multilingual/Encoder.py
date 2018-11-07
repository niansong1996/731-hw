from typing import List, Tuple

import torch
from utils import assert_tensor_size
from FLSTM import Stack_FLSTMCell
from torch import Tensor
from config import device
import torch.nn.functional as F


class Encoder(torch.nn.Module):
    """
    The encoder is a bidiretional encoder, one can NOT be used as a single direction one
    """

    def __init__(self, batch_size, embed_size, hidden_size, training, dropout_rate, num_layer=2):
        self.num_direction = 2
        # init size constant
        self.batch_size = batch_size
        self.input_size = embed_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer

        # set different layers
        self.embed_size = embed_size
        self.in_order_cells = Stack_FLSTMCell(self.input_size, self.hidden_size, num_layers=num_layer)
        self.rev_order_cells = Stack_FLSTMCell(self.input_size, self.hidden_size, num_layers=num_layer)
        self.training = training
        self.dropout_rate = dropout_rate

        # set some dummy input states
        self.h_0 = torch.zeros((self.num_direction * self.num_layer, self.batch_size, self.hidden_size), device=device)
        self.c_0 = torch.zeros((self.num_direction * self.num_layer, self.batch_size, self.hidden_size), device=device)

    def __call__(self, src_sent_idx: Tensor, word_embedding) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        encode the sequence in bidirection

        Args:
            src_sent_idx: source sentence word indices dim = (batch_size, sent_len)

        Return:
            outputs: dim = (sent_length, batch_size, num_direction * hidden_size)
            h_t, c_t: dim = (num_layers, batch_size, num_direction * hidden_size)
        """
        # set the variables that iteratively used in encoding steps
        h_t = self.h_0
        c_t = self.c_0
        outputs = []

        # get the sentence length for this batch
        sent_len = src_sent_idx.shape[1]

        # dim = (batch_size, sent_length, embed_size)
        embedding = word_embedding(src_sent_idx)

        # for each of the sent words, encode step by step
        for step in range(sent_len):
            output, h_t, c_t = self.encoder_step(embedding[:, step, :], embedding[:, -step - 1, :], h_t, c_t)
            outputs.append(output)

        # pack the list of tensors to one single tensor
        outputs = torch.stack(outputs, dim=0)

        return outputs, (self.to_tensor(h_t), self.to_tensor(c_t))

    def to_tensor(self, t: List[Tensor]) -> Tensor:
        """
        Concatenates a list of tensors with two directions and stack by layers
        :param t:
        :return: dim = (num_layers, batch_size, num_direction * hidden_size)
        """
        # dim = (num_layers, batch_size, hidden_size)
        in_t = torch.stack(t[:self.num_layer], dim=0)
        # dim = (num_layers, batch_size, hidden_size)
        rev_t = torch.stack(t[self.num_layer:], dim=0)
        # dim = (num_layers, batch_size, num_direction * hidden_size)
        return torch.cat((in_t, rev_t), 2)

    def encoder_step(self, in_x: Tensor, rev_x: Tensor, h_t: List[Tensor], c_t: List[Tensor]) \
            -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        """
        encode only one step by two direction for stacked flstm
        Args:
            in_x: in-order input dim = (batch_size, input_size)
            rev_x: reverse-order input dim = (batch_size, input_size)
            h_t: a list of num_direction * num_layers previous hidden layers of shape [batch_size, hidden_size]
            c_t: a list of num_direction * num_layers previous memory cells of shape [batch_size, hidden_size]

        Return:
            output: the hidden state of only the last layer concat with directions,
                    dim = (batch_size, num_direction * hidden_size)
            h_t_1: a list of num_direction * num_layers next hidden layers of shape [batch_size, hidden_size]
            c_t_1: a list of num_direction * num_layers next memory cells of shape [batch_size, hidden_size]
        """
        assert (len(h_t) == self.num_direction * self.num_layer)
        assert (len(c_t) == self.num_direction * self.num_layer)
        in_x = F.dropout(in_x, p=self.dropout_rate, training=self.training)
        # one step encode for both directions
        # dim = (num_layers, batch_size, hidden_size)
        in_h_t_1, in_c_t_1 = self.in_order_cells(in_x, h_t[:self.num_layer], c_t[:self.num_layer])
        rev_h_t_1, rev_c_t_1 = self.rev_order_cells(rev_x, h_t[self.num_layer:], c_t[self.num_layer:])
        # dim = (batch_size, hidden_size)
        in_output = in_h_t_1[-1]
        rev_output = rev_h_t_1[-1]

        output = torch.cat([in_output, rev_output], dim=1)

        h_t_1 = in_h_t_1 + rev_h_t_1
        c_t_1 = in_c_t_1 + rev_c_t_1

        return output, h_t_1, c_t_1
