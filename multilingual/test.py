import torch
import torch.nn as nn
import torch.tensor as Tensor
import torch.nn.functional as F

class CPG(nn.Module):
    def __init__(self, P, M, L):
        super(CPG, self).__init__()

        self.W = nn.Linear(M, P, bias=False)
        self.L_embed = nn.Linear(L, M, bias=False)

        self.lstm = nn.LSTM(100, 300)

        self.model = nn.Linear(P, 1, bias=False)

        self.criterion = nn.MSELoss()

        # initialize the parameters using uniform distribution
        for param in self.named_parameters():
            print(param[0])
            nn.init.uniform_(param[1].data, a=-0.1, b=0.1)

    def forward(self, L, X, y):
        # dim(L) = [L, 1]
        # dim(x) = [P, 1]

        L = self.L_embed(L) # L = [M, 1]
        theta = self.W(L) # theta = [P, 1]

        theta = nn.Parameter(theta, requires_grad=True)

        self.model.weight = nn.Parameter(theta, requires_grad=True)
        hyp_y = self.model(X.transpose(0,1))

        loss = self.criterion(hyp_y, y)

        return loss 

P = 10
M = 3
L = 2 
cpg = CPG(P, M, L)
optimizer = torch.optim.Adam(cpg.parameters(), lr=0.01)

for param in cpg.named_parameters():
    print(param.shape)

x = torch.randn((P, 1))
y = torch.randn((1,1))
l = torch.tensor([[1, 0]], dtype=torch.float)
exit(0)
for _ in range(100):
    optimizer.zero_grad()
    loss = cpg(l, x, y)
    for param in cpg.parameters():
        print(param.grad)
    loss.backward()
    optimizer.step()

    print(loss)
    
    #print(cpg.W.weight)
