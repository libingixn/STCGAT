# -*- coding: utf-8 -*-
# @Project: STCGAT
# @Author  : shiqiZhang
import torch
import torch.nn as nn
from models.GAT import GATSubNet
from models.Tcn import TCN


class STCGAT(nn.Module):
    def __init__(self, device, input_dim, gat_units, gatOut_dim, gat_heads, dropout, predict_length,
                 lstm_units, num_layers, alpha, tcn_units, d, kernel_size):
        super(STCGAT, self).__init__()
        self.device = device
        self.dropout = dropout
        self.predict_length = predict_length
        self.num_layers = num_layers
        self.hidden_size = lstm_units

        self.subnet = GATSubNet(input_dim, gat_units, gatOut_dim, gat_heads, alpha, dropout)
        self.lstm = nn.LSTM(gatOut_dim, lstm_units, num_layers, bidirectional=True)
        self.tcn = TCN(num_inputs=lstm_units*2, tcn_hidden=tcn_units, predict_length=predict_length, d=d, kernel_size=kernel_size)

    def forward(self, x, adj):

        gat_output = self.subnet(x, adj)
        h0 = torch.randn(self.num_layers * 2, x.size(1), self.hidden_size).to(self.device)
        c0 = torch.randn(self.num_layers * 2, x.size(1), self.hidden_size).to(self.device)

        output, (hn, cn) = self.lstm(gat_output, (h0, c0))

        out = self.tcn(output)

        return out