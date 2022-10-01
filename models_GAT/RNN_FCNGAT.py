#export
from tsai.imports import *
from tsai.models.layers import *


class _RNN_FCN_BaseGAT(Module):
    def __init__(self, c_in, c_out, seq_len=None, hidden_size=100, rnn_layers=1, bias=True, cell_dropout=0,
                 rnn_dropout=0.8, bidirectional=False, shuffle=True,
                 fc_dropout=0., conv_layers=[128, 256, 128], kss=[7, 5, 3], se=0):
        if shuffle: assert seq_len is not None, 'need seq_len if shuffle=True'

        # RNN
        self.rnn = self._cell(seq_len if shuffle else c_in, hidden_size, num_layers=rnn_layers, bias=bias,
                              batch_first=True,
                              dropout=cell_dropout, bidirectional=bidirectional)
        self.rnn_dropout = nn.Dropout(rnn_dropout) if rnn_dropout else noop
        self.shuffle = Permute(0, 2,
                               1) if not shuffle else noop  # You would normally permute x. Authors did the opposite.

        # FCN
        assert len(conv_layers) == len(kss)
        self.convblock1 = ConvBlock(c_in, conv_layers[0], kss[0])
        self.se1 = SqueezeExciteBlock(conv_layers[0], se) if se != 0 else noop
        self.convblock2 = ConvBlock(conv_layers[0], conv_layers[1], kss[1])
        self.se2 = SqueezeExciteBlock(conv_layers[1], se) if se != 0 else noop
        self.convblock3 = ConvBlock(conv_layers[1], conv_layers[2], kss[2])
        self.gap = GAP1d(1)

        # Common
        self.concat = Concat()
        self.fc_dropout = nn.Dropout(fc_dropout) if fc_dropout else noop
        self.fc = nn.Linear(hidden_size * (1 + bidirectional) + conv_layers[-1], c_out)

        self.transformation = nn.Linear((hidden_size * (1 + bidirectional) + conv_layers[-1]),(hidden_size * (1 + bidirectional) + conv_layers[-1]))
        self.a = nn.Parameter(torch.randn((hidden_size * (1 + bidirectional) + conv_layers[-1]) * 2, 1))
        self.a.requires_grad = True
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

        self.gnn_fc_out = nn.Sequential(
                                nn.Linear((hidden_size * (1 + bidirectional) + conv_layers[-1]),(hidden_size * (1 + bidirectional) + conv_layers[-1])),
                                nn.LeakyReLU(),
                                nn.Linear((hidden_size * (1 + bidirectional) + conv_layers[-1]),c_out),
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
        # RNN
        rnn_input = self.shuffle(x)  # permute --> (batch_size, seq_len, n_vars) when batch_first=True
        output, _ = self.rnn(rnn_input)
        last_out = output[:, -1]  # output of last sequence step (many-to-one)
        last_out = self.rnn_dropout(last_out)

        # FCN
        x = self.convblock1(x)
        x = self.se1(x)
        x = self.convblock2(x)
        x = self.se2(x)
        x = self.convblock3(x)
        x = self.gap(x)

        # Concat
        x = self.concat([last_out, x])
        hidden = self.fc_dropout(x)
        origin_out = self.fc(hidden)

        att_weight = self.cal_attention(hidden)
        hidden = att_weight.mm(hidden) + hidden
        gat_out = self.gnn_fc_out(hidden)
        return {
            "origin_out": origin_out,
            "gat_out": gat_out,
            "x_embeddings": hidden
        }

class RNN_FCNGAT(_RNN_FCN_BaseGAT):
    _cell = nn.RNN


class LSTM_FCNGAT(_RNN_FCN_BaseGAT):
    _cell = nn.LSTM


class GRU_FCNGAT(_RNN_FCN_BaseGAT):
    _cell = nn.GRU


class MRNN_FCNGAT(_RNN_FCN_BaseGAT):
    _cell = nn.RNN

    def __init__(self, *args, se=16, **kwargs):
        super().__init__(*args, se=16, **kwargs)


class MLSTM_FCNGAT(_RNN_FCN_BaseGAT):
    _cell = nn.LSTM

    def __init__(self, *args, se=16, **kwargs):
        super().__init__(*args, se=16, **kwargs)


class MGRU_FCNGAT(_RNN_FCN_BaseGAT):
    _cell = nn.GRU

    def __init__(self, *args, se=16, **kwargs):
        super().__init__(*args, se=16, **kwargs)


if __name__ == '__main__':
    a = torch.ones((128, 1, 35))
    model = LSTM_FCNGAT(1, 4,shuffle = False)
    out = model(a)
    for key in out:
        if out[key] is not None:
            print(out[key].shape)