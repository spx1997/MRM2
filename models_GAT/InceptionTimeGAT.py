from tsai.models.layers import *
from tsai.all import *
from tsai.utils import *

import torch.nn as nn
import torch

class InceptionTimeGAT(Module):
    def __init__(self,c_in,c_out,seq_len= None,nf = 32,nb_filters = None, **kwargs):
        nf = ifnone(nf,nb_filters)
        self.inceptionblock = InceptionBlock(c_in, nf, **kwargs)
        self.gap = GAP1d(1)
        self.fc = nn.Linear(nf * 4, c_out)

        self.transformation = nn.Linear(nf * 4,nf * 4)
        self.a = nn.Parameter(torch.randn(nf * 4 * 2, 1))
        self.a.requires_grad = True
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

        self.gnn_fc_out = nn.Sequential(
                                nn.Linear(nf * 4,nf * 4),
                                nn.LeakyReLU(),
                                nn.Linear(nf * 4,c_out),
                                )
    def cal_attention(self,x):
        x1 = self.transformation(x)

        sample_num,dim = x1.shape
        e_x = x1.expand(sample_num,sample_num,dim)
        e_y = torch.transpose(e_x,0,1)
        attention_in = torch.cat((e_x, e_y), 2).view(-1, dim * 2)
        self.a_t = torch.t(self.a)
        attention_out = self.a_t.mm(torch.t(attention_in)).view(sample_num, sample_num)
        attention_out = self.leaky_relu(attention_out)
        att_weight = self.softmax(attention_out)
        return att_weight


    def forward(self, x):
        x = self.inceptionblock(x)
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
    model = InceptionTimeGATs(1, 4)
    out = model(a)
    for key in out:
        if out[key] is not None:
            print(out[key].shape)