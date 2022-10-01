from tsai.all import *
import torch


class ResNetMRM2(Module):
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

    def forward(self, x, used_ttrm2=False,alpha = 1,beta = 1):
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.squeeze(self.gap(x))
        tcrm2_out = self.fc(x)
        ttrm2_out = None

        if used_ttrm2:
            tcrm2_softmax_out = torch.softmax(tcrm2_out, dim=1) ** alpha
            similar_x_x = torch.matmul(x, x.T)
            similar_x_x_stand = similar_x_x * beta /( similar_x_x.std(1, keepdim=True)+1e-8)
            ttrm2_out = torch.matmul((similar_x_x_stand),
                                     tcrm2_softmax_out /( tcrm2_softmax_out.sum(dim=0, keepdim=True)+1e-8))

        return {
            "tcrm2_out":tcrm2_out,
            "ttrm2_out":ttrm2_out,
            "x_embeddings":x
        }
if __name__ == '__main__':
    a = torch.ones((128, 1, 35))
    model = ResNetMRM2(1, 4)
    out = model(a,torch.randint(0,2,(128,))*-1)
    for key in out:
        if out[key] is not None:
            print(out[key].shape)