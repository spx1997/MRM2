from random import sample
from tkinter import Label
from tsai.imports import *
from tsai.models.layers import *
import torch.nn as nn
import torch


class FCNGAT(Module):
    def __init__(self, c_in, c_out, layers=[128, 256, 128], kss=[7, 5, 3]):
        assert len(layers) == len(kss)
        self.convblock1 = ConvBlock(c_in, layers[0], kss[0])
        self.convblock2 = ConvBlock(layers[0], layers[1], kss[1])
        self.convblock3 = ConvBlock(layers[1], layers[2], kss[2])
        self.gap = GAP1d(1)
        self.fc = nn.Linear(layers[-1], c_out)

        self.transformation = nn.Linear(layers[-1], layers[-1])
        self.a = nn.Parameter(torch.randn(layers[-1] * 2, 1))
        self.a.requires_grad = True
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

        self.gnn_fc_out = nn.Sequential(
            nn.Linear(layers[-1], layers[-1]),
            nn.LeakyReLU(),
            nn.Linear(layers[-1], c_out),
        )

    def cal_attention(self, x):
        x1 = self.transformation(x)
        sample_num, dim = x1.shape
        e_x = x1.expand(sample_num, sample_num, dim)
        e_y = torch.transpose(e_x, 0, 1)
        attention_in = torch.cat((e_x, e_y), 2).view(-1, dim * 2)
        self.a_t = torch.t(self.a)
        attention_out = self.a_t.mm(torch.t(attention_in)).view(sample_num, sample_num)
        attention_out = self.leaky_relu(attention_out)
        att_weight = self.softmax(attention_out)
        return att_weight

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        hidden = self.gap(x)
        origin_out = self.fc(hidden)
        att_weight = self.cal_attention(hidden)
        hidden = att_weight.mm(hidden) + hidden
        gat_out = self.gnn_fc_out(hidden)
        return {
            "origin_out": origin_out,
            "gat_out": gat_out,
            "x_embeddings": hidden
        }


if __name__ == '__main__':
    a = torch.ones((64, 1, 35))
    model = FCNGATs(1, 4)
    out = model(a)
    for key in out:
        if out[key] is not None:
            print(out[key].shape)
