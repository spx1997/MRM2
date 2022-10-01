from tsai.models.layers import *
from tsai.all import *
from tsai.utils import *

class InceptionTime(Module):
    def __init__(self,c_in,c_out,seq_len= None,nf = 32,nb_filters = None, **kwargs):
        nf = ifnone(nf,nb_filters)
        self.inceptionblock = InceptionBlock(c_in, nf, **kwargs)
        self.gap = GAP1d(1)
        self.fc = nn.Linear(nf * 4, c_out)

    def forward(self, x):
        x = self.inceptionblock(x)
        x = self.gap(x)
        origin_out = self.fc(x)
        return {
            "origin_out": origin_out,
            "x_embeddings": x
        }

if __name__ == '__main__':
    a = torch.ones((128, 1, 35))
    model = InceptionTime(1, 4)
    out = model(a)
    for key in out:
        if out[key] is not None:
            print(out[key].shape)