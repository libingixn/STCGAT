# -*- coding: utf-8 -*-
# @Project: STCGAT
# @Author  : shiqiZhang
# @Time    : 2022/2/26 14:49
# @Phrase: 人生不如意十之八九，Bug是一二。
import torch
import torch.nn as nn
from layers.GraphAttentionLayer import GraphAttentionLayer


class GATSubNet(nn.Module):
    def __init__(self, history_length, gat_hidden, gat_output, gat_heads, alpha, dropout):
        super(GATSubNet, self).__init__()
        self.dropout = dropout
        # Use loops to add more attention, and use nn.ModuleList to turn it into a large parallel network
        self.attention_module = nn.ModuleList(
            [GraphAttentionLayer(history_length, gat_hidden, dropout) for _ in range(gat_heads)])

        self.out_att = GraphAttentionLayer(gat_hidden * gat_heads, gat_output, dropout)

        self.act = nn.LeakyReLU(alpha)

    def forward(self, x, graph):
        """
        :param inputs: [B, N, C]
        :param graph: [N, N]
        :return:
        """
        B, N = x.size(0), x.size(1)
        x = x.view(B, N, -1)  # [B, N, H*D]

        # inputs = F.dropout(inputs, self.dropout, training=self.training)
        outputs = torch.cat([attn(x, graph) for attn in self.attention_module], dim=-1)  # [B, N, hid_c * h_head]
        # outputs = F.dropout(outputs, self.dropout, training=self.training)
        outputs = self.act(outputs)

        outputs = self.out_att(outputs, graph)

        return self.act(outputs)