from tsai.all import *
import torch


class ResNet(Module):
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

    def forward(self, x):
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.squeeze(self.gap(x))
        origin_out = self.fc(x)
        return {
            "origin_out": origin_out,
            "x_embeddings": x
        }
if __name__ == '__main__':
    a = torch.ones((128, 1, 35))
    model = ResNet(1, 4)
    out = model(a)
    for key in out:
        if out[key] is not None:
            print(out[key].shape)