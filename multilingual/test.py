import torch
import torch.nn as nn
import torch.tensor as Tensor
import torch.nn.functional as F

class CPG(nn.Module):
    def __init__(self, P, M, L):
        super(CPG, self).__init__()

        self.W = nn.Linear(M, P, bias=False)
        self.L_embed = nn.Linear(L, M, bias=False)
        # self.Linear = nn.Linear(P, 1, bias=False)
        self.criterion = nn.MSELoss()
        self.encoder_lstm = nn.LSTM(3, 4, num_layers=5, bidirectional=False)

        # initialize the parameters using uniform distribution
        for param in self.named_parameters():
            print(param[0])
            nn.init.uniform_(param[1].data, a=0.5, b=0.5)

    def forward(self, L, X, y):
        # dim(L) = [L, 1]
        # dim(x) = [P, 1]

        L = self.L_embed(L) # L = [M, 1]
        # print(L)
        theta = self.W(L) # theta = [P, 1]
        # print(theta)

        # hyp_y = F.linear(X.transpose(0,1), theta)
        # self.Linear.weight.data = theta


        decoder = Decoder(theta)
        hyp_y = decoder.decode(X.transpose(0, 1))

        #hyp_y = F.linear(X.transpose(0, 1), theta)

        loss = self.criterion(hyp_y, y)

        return loss

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.encoder_lstm = nn.LSTM(3, 4, num_layers=1, bidirectional=False)
        for param in self.named_parameters():
            nn.init.uniform_(param[1].data, a=-0.1, b=0.1)

    def forward(self, y):
        return y


class Decoder:
    def __init__(self, Ws):
        self.Ws = Ws

    def decode(self, x):
        return F.linear(x, self.Ws)


P = 10
M = 3
L = 2 
cpg = CPG(P, M, L)
optimizer = torch.optim.Adam(cpg.parameters(), lr=0.01)


# for param in LSTM().parameters():
#     print(param)
#     print("parameter shape: %s" % param.dim())

x = torch.ones((P, 1))
y = torch.ones((1,1))
l = torch.tensor([[1, 0]], dtype=torch.float)

for _ in range(10):
    optimizer.zero_grad()
    loss = cpg(l, x, y)
    # for param in cpg.parameters():
    #     print(param.grad)
    loss.backward()
    optimizer.step()

    print(loss.data, end='')

    #print(cpg.W.weight)
