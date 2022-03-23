# -*- coding: utf-8 -*-
# @Project: STCGAT
# @Author  : shiqiZhang
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_c, out_c, dropout):
        super(GraphAttentionLayer, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.dropout = dropout
        self.F = F.softmax

        self.W = nn.Linear(in_c, out_c, bias=False)  # y = W * x
        self.b = nn.Parameter(torch.Tensor(out_c))

        nn.init.normal_(self.W.weight)
        nn.init.normal_(self.b)

    def forward(self, inputs, graph):
        """
        :param inputs: input features, [B, N, C].
        :param graph: graph structure, [N, N].
        :return:
            output features, [B, N, D].
        """
        h = self.W(inputs)  # [B, N, D]

        outputs = torch.bmm(h, h.transpose(1, 2)) * graph.unsqueeze(
            0)  # [B, N, D]*[B, D, N]->[B, N, N],         x(i)^T * x(j)

        # Since 0 in the result calculated above indicates that it is okay between the nodes, these 0s are replaced with negative infinity, since negative infinity of softmax = 0
        outputs.data.masked_fill_(torch.eq(outputs, 0), -float(1e16))

        attention = self.F(outputs, dim=2)  # [B, N, N]，The attention factor is obtained by normalizing the nodes in dimension 2, which means that all nodes connected by edges are normalized.
        # attention = F.dropout(attention, self.dropout, training=self.training)

        return torch.bmm(attention, h) + self.b  # [B, N, N] * [B, N, D] Differentiated information aggregation of neighboring nodes using attention coefficients