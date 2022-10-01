from tsai.imports import *
from tsai.models.layers import *


class FCNMRM2(Module):
    def __init__(self, c_in, c_out, layers=[128, 256, 128], kss=[7, 5, 3]):
        assert len(layers) == len(kss)
        self.convblock1 = ConvBlock(c_in, layers[0], kss[0])
        self.convblock2 = ConvBlock(layers[0], layers[1], kss[1])
        self.convblock3 = ConvBlock(layers[1], layers[2], kss[2])
        self.gap = GAP1d(1)
        self.fc = nn.Linear(layers[-1], c_out)

    def forward(self, x, used_ttrm2=False, alpha=1, beta=1):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.gap(x)
        tcrm2_out = self.fc(x)
        ttrm2_out = None

        if used_ttrm2:
            tcrm2_softmax_out = torch.softmax(tcrm2_out, dim=1) ** alpha
            similar_x_x = torch.matmul(x, x.T)
            similar_x_x_stand = similar_x_x * beta/( similar_x_x.std(1, keepdim=True)+1e-8)
            ttrm2_out = torch.matmul(similar_x_x_stand, tcrm2_softmax_out / (tcrm2_softmax_out.sum(dim=0, keepdim=True)+1e-8))
        return {
            "tcrm2_out": tcrm2_out,
            "ttrm2_out": ttrm2_out,
            "x_embeddings": x
        }


if __name__ == '__main__':
    a = torch.ones((128, 1, 35))
    model = FCNMRM2(1, 4)
    out = model(a, torch.randint(0, 2, (128,)) * -1)
    for key in out:
        if out[key] is not None:
            print(out[key].shape)
