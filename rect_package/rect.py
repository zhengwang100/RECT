import torch
import torch.nn as nn
import torch.nn.functional as F
from rect_package.gcn import GCN

class RECT(nn.Module):
    def __init__(self, n_in, n_h, activation, dropout=0.5):
        super(RECT, self).__init__()
        self.gcn_1 = GCN(n_in, n_h, activation)
        self.mlp = nn.Linear(n_h, n_in)
        self.dropout = dropout
        torch.nn.init.xavier_uniform_(self.mlp.weight.data)

    def forward(self, seq1, adj, sparse):
        h_1 = self.gcn_1(seq1, adj, sparse)
        h_1 = F.dropout(h_1, p=self.dropout, training=self.training)
        preds = self.mlp(h_1) # predict the class-center
        return h_1, preds

    # Detach the return variables
    def embed(self, seq, adj, sparse):
        h_1 = self.gcn_1(seq, adj, sparse) 
        return h_1.detach()