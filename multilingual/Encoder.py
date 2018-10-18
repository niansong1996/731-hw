from typing import List

import torch
from utils import assert_tensor_size
from FLSTM import Stack_FLSTMCell
from torch import Tensor
import torch.nn.functional as F
from config import device


class Encoder:
    """
    The encoder is a bidiretional encoder, one can NOT be used as a single direction one
    """
    def __init__(self, batch_size, input_size, hidden_size, embedding_weights: Tensor, weights: Tensor, num_layer=2):
        self.num_direction = 2
        # num of cell weights must match the setting
        assert(len(weights) == num_layer * self.num_direction)
        # init size constant
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer

        # set different layers
        self.embedding_weights = embedding_weights
        self.embed_size = embedding_weights.shape[1]
        self.in_order_cells = Stack_FLSTMCell(self.input_size, self.hidden_size, weights[:self.num_layer])
        self.rev_order_cells = Stack_FLSTMCell(self.input_size, self.hidden_size, weights[self.num_layer:])

        # set some dummy input states
        self.h_0 = torch.zeros((self.num_direction * self.num_layer, self.batch_size, self.hidden_size), device=device)
        self.c_0 = torch.zeros((self.num_direction * self.num_layer, self.batch_size, self.hidden_size), device=device)

    def __call__(self, src_idx: Tensor) -> Tensor:
        """
        encode the sequence in bidirection

        Args:
            src_idx: source input indices dim = (batch_size, sent_len)

        Return:
            outputs: dim = (sent_length, batch_size, num_direction * hidden_size)
        """
        # set the variables that iteratively used in encoding steps
        h_t = self.h_0
        c_t = self.c_0
        outputs = []

        # get the sentence length for this batch
        sent_len = src_idx.shape[1]

        # dim = (batch_size, sent_length, embed_size)
        embedding = F.embedding(src_idx, self.embedding_weights)
        # dim = (batch_size, sent_length, embed_size)
        assert_tensor_size(embedding, [self.batch_size, sent_len, self.embed_size])

        # for each of the sent words, encode step by step
        for step in range(sent_len):
            output, h_t, c_t = self.encoder_step(embedding[:, step, :], embedding[:, -step-1, :], h_t, c_t)
            outputs.append(output)

        # pack the list of tensors to one single tensor
        outputs = torch.stack(outputs, dim=0)
        assert_tensor_size(outputs, [sent_len, self.batch_size, self.num_direction * self.hidden_size])

        return outputs

    def encoder_step(self, in_x: Tensor, rev_x: Tensor, h_t: List[Tensor], c_t: List[Tensor]) \
            -> (Tensor, List[Tensor], List[Tensor]):
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
        assert(len(h_t) == self.num_direction * self.num_layer)
        assert(len(c_t) == self.num_direction * self.num_layer)
        assert_tensor_size(in_x, [self.batch_size, self.embed_size])
        assert_tensor_size(rev_x, [self.batch_size, self.embed_size])

        # one step encode for both directions
        # dim = (num_layers, batch_size, hidden_size)
        in_h_t_1, in_c_t_1 = self.in_order_cells(in_x, h_t[:self.num_layer], c_t[:self.num_layer])
        rev_h_t_1, rev_c_t_1 = self.in_order_cells(rev_x, h_t[self.num_layer:], c_t[self.num_layer:])
        # dim = (batch_size, hidden_size)
        in_output = in_h_t_1[-1]
        rev_output = rev_h_t_1[-1]

        output = torch.stack((in_output, rev_output), dim=1)
        assert_tensor_size(output, [self.batch_size, 2 * self.hidden_size])
        h_t_1 = in_h_t_1 + rev_h_t_1
        c_t_1 = in_c_t_1 + rev_c_t_1

        return output, h_t_1, c_t_1




