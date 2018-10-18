import torch
import torch.nn as nn
from FLSTM import FLSTMCell, Stack_FLSTMCell
from torch import Tensor

class Encoder():
    '''
    The encoder is a bidiretional encoder, one can NOT be used as a single direction one
    '''
    def __init__(self, batch_size, input_size, hidden_size, word_embedding, weights, num_layer=2):
        # num of cell weights must match the setting
        assert(len(weights) == num_layer * 2)

        # init size constant
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = 2 
        self.num_layer = num_layer
        self.bidirectional = bidirectional

        # set different layers
        self.embed = word_embedding
        self.in_order_cells = Stack_FLSTMCell(self.input_size, self.hidden_size, weights[:self.num_layer])
        self.rev_order_cells = Stack_FLSTMCell(self.input_size, self.hidden_size, weights[self.num_layer:])

        # set some dummy input states
        self.h_0 = torch.zeros((self.num_direction * self.num_layer, self.batch_size, self.hidden_size)) 
        self.c_0 = torch.zeros((self.num_direction * self.num_layer, self.batch_size, self.hidden_size)) 

    
    def __call__(self, src_encoding):
        '''
        encode the sequence in bidirection

        Args:
            src_encoding: dim = (sent_length, batch_size, input_size)
        
        Return:
            outputs: dim = (sent_length, batch_size, num_direction * hidden_size)
        '''
        # get the sentence length of this batch
        sent_length = len(src_encoding)

        # set the variables that iteratively used in encoding steps
        h_t = self.h_0
        c_t = self.c_0
        outputs = []

        # dim = (sent_length, batch_size, input_size)
        embedding = self.embed(src_encoding)

        # for each of the sent words, encode step by step
        for step in range(sent_length): 
            output, h_t, c_t = self.encoder_step(packed_seqs)
            outputs.append(output)

        # pack the list of tensors to one single tensor
        outputs = torch.stack(outputs, dim=0)

        return outputs


    def encoder_step(self, X, h_t, c_t):
        '''
        encode only one step by two direction for stacked flstm
        Args:
            X: dim = (batch_size, input_size)
            h_t: dim = (num_direction * num_layers, batch_size, hidden_size)
            c_t: dim = (num_direction * num_layers, batch_size, hidden_size)

        Return:
            output: the hidden state of only the last layer concat with directions, 
                    dim = (batch_size, num_direction * hidden_size)
            h_t_1: next hidden state of all directions and all layers,
                    dim = (num_direction * num_layers, batch_size, hidden_size)
            c_t_1: next cell state of all directions and all layers
                    dim = (num_direction * num_layers, batch_size, hidden_size)
        '''
        assert(len(h_t) == self.num_direction * self.num_layer)
        assert(len(c_t) == self.num_direction * self.num_layer)
        assert(len(X) == self.num_direction)

        # one step encode for both directions
        # dim = (num_layers, batch_size, hidden_size)
        in_h_t_1, in_c_t_1 = self.in_order_cells(X[0], h_t[:self.num_layer], c_t[:self.num_layer])
        rev_h_t_1, rev_c_t_1 = self.in_order_cells(X[0], h_t[self.num_layer:], c_t[self.num_layer:])
        # dim = (batch_size, hidden_size)
        in_output = in_h_t_1[-1]
        rev_output = rev_h_t_1[-1]

        output = torch.stack((in_output, rev_output), dim=1)
        h_t_1 = in_h_t_1 + rev_h_t_1
        c_t_1 = in_c_t_1 + rev_c_t_1

        return output, h_t_1, c_t_1




