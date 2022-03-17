# -*- coding: utf-8 -*-
# @Project: STCGAT
# @Author  : shiqiZhang
# @Time    : 2022/2/26 14:48
# @Phrase: 人生不如意十之八九，Bug是一二。
import torch
import torch.nn as nn
from models.GAT import GATSubNet
from models.Tcn import TCN


class STCGAT(nn.Module):
    def __init__(self, device, nfeat, nhid, gatOut, nheads, dropout, predict_length,
                 hidden_size, num_layers, alpha, tcn_hidden, d, kernel_size):
        super(STCGAT, self).__init__()
        self.device = device
        self.dropout = dropout
        self.predict_length = predict_length
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.subnet = GATSubNet(nfeat, nhid, gatOut, nheads, alpha, dropout)
        self.lstm = nn.LSTM(gatOut, hidden_size, num_layers, bidirectional=True)
        self.tcn = TCN(num_inputs=hidden_size*2, tcn_hidden = tcn_hidden, predict_length=predict_length, d=d, kernel_size=kernel_size)


    def forward(self, x, adj):

        gat_output = self.subnet(x, adj)
        h0 = torch.randn(self.num_layers * 2, x.size(1), self.hidden_size).to(self.device)
        c0 = torch.randn(self.num_layers * 2, x.size(1), self.hidden_size).to(self.device)

        output, (hn, cn) = self.lstm(gat_output, (h0, c0))

        out = self.tcn(output)

        return out