import torch
import torch.nn as nn
import torch.nn.functional as F

def unpack_weight(self, weight, input_size, hidden_size):
    assert(weight.shape == (4*hidden_size*(input_size+hidden_size+1+1), 1))

    W_x_shape = (4*hidden_size, input_size)
    W_h_shape = (4*hidden_size, hidden_size)
    b_x_shape = (4*hidden_size, 1)
    b_h_shape = (4*hidden_size, 1)

    W_x_0 = 0
    W_x_1 = W_x_shape[0] * W_x_shape[1]
    W_h_1 = W_x_1 + W_h_shape[0] * W_h_shape[1]
    b_x_1 = W_h_1 + b_x_shape[0]
    b_h_1 = b_x_1 + b_h_shape[0]

    W_x = weight[W_x_0:W_x_1].reshape(W_x_shape)
    W_h = weight[W_x_1:W_h_1].reshape(W_h_shape)
    b_x = weight[W_h_1:b_x_1].reshape(b_x_shape)
    b_h = weight[b_x_1:b_h_1].reshape(b_h_shape)

    return W_x, W_h, b_x, b_h 

class FLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

    
    def forward(self, X, h_0, c_0, weights):
        """
        perform a step in LSTM

        :param X: input embedding dim = (batch_size, embed_size) 
        :param c_0: cell state (batch_size, hidden_size)
        :param h_0: hidden state (batch_size, hidden_size)
        :param weights: W_x, W_h, b_x, b_h

        :return: (h_1, c_1) next cell state and hidden state
        """
        batch_size = X.shape[0]
        input_size = X.shape[1]
        hidden_size = h_0.shape[1]

        W_x, W_h, b_x, b_h = weights

        # (4*hidden_size, batch_size)  =  (4*hidden_size, input_size) * (batch_size, input_size)^{T}
        W_x_X = torch.mm(W_x, X.transpose(0 ,1))
        W_h_H = torch.mm(W_h, h_0.transpose(0,1))
        W_x_h = W_x_X + W_h_H
        W_x_h_b = W_x_h + b_x + b_h

        # (batch_size, hidden_size)
        i = torch.sigmoid(W_x_h_b[0:hidden_size]).transpose(0,1)
        f = torch.sigmoid(W_x_h_b[hidden_size:2*hidden_size]).transpose(0,1)
        g = torch.tanh(W_x_h_b[2*hidden_size:3*hidden_size]).transpose(0,1)
        o = torch.sigmoid(W_x_h_b[3*hidden_size:4*hidden_size]).transpose(0,1)

        # (batch_size, hidden_size)
        c_1 = f * c_0 + i * g
        h_1 = o * torch.tanh(c_1)

        return h_1, c_1

    
input_size = 256
hidden_size = 512
batch_size = 32

c_0 = torch.randn((batch_size, hidden_size))
h_0 = torch.randn((batch_size, hidden_size))
X = torch.randn((batch_size, input_size))

lstm = nn.LSTMCell(input_size, hidden_size)
flstm = FLSTM(input_size, hidden_size)

weights = []
for param in lstm.parameters():
    weights.append(param.data)
weights[2] = weights[2].unsqueeze(1)
weights[3] = weights[3].unsqueeze(1)

lstm_output = lstm(X, (h_0, c_0))
flstm_output = flstm(X, h_0, c_0, weights)

print(lstm_output)
print(flstm_output)


        






