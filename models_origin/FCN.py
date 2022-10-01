from tsai.imports import *
from tsai.models.layers import *

class FCN(Module):
    def __init__(self, c_in, c_out, layers=[128, 256, 128], kss=[7, 5, 3]):
        assert len(layers) == len(kss)
        self.convblock1 = ConvBlock(c_in, layers[0], kss[0])
        self.convblock2 = ConvBlock(layers[0], layers[1], kss[1])
        self.convblock3 = ConvBlock(layers[1], layers[2], kss[2])
        self.gap = GAP1d(1)
        self.fc = nn.Linear(layers[-1], c_out)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.gap(x)
        origin_out = self.fc(x)

        return {
            "origin_out": origin_out,
            "x_embeddings": x
        }
if __name__ == '__main__':
    a = torch.ones((128, 1, 35))
    model = FCN(1, 4)
    out = model(a)
    for key in out:
        if out[key] is not None:
            print(out[key].shape)