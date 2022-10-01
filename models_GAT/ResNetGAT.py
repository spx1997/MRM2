from tsai.all import *
import torch


class ResNetGAT(Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        nf = 64
        kss = [7, 5, 3]
        self.c_out = c_out
        self.resblock1 = ResBlock(c_in, nf, kss=kss)
        self.resblock2 = ResBlock(nf, nf * 2, kss=kss)
        self.resblock3 = ResBlock(nf * 2, nf * 2, kss=kss)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.squeeze = Squeeze(-1)
        self.fc = nn.Linear(nf * 2, c_out)

        self.transformation = nn.Linear(nf * 2, nf * 2)
        self.a = nn.Parameter(torch.randn(nf * 2 * 2, 1))
        self.a.requires_grad = True
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

        self.gnn_fc_out = nn.Sequential(
            nn.Linear(nf * 2, nf * 2),
            nn.LeakyReLU(),
            nn.Linear(nf * 2, c_out),
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
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        hidden = self.squeeze(self.gap(x))
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
    a = torch.ones((128, 1, 35))
    model = ResNetGATs(1, 4)
    out = model(a)
    for key in out:
        if out[key] is not None:
            print(out[key].shape)