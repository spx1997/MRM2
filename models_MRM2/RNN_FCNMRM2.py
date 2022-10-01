#export
from tsai.imports import *
from tsai.models.layers import *


class _RNN_FCN_BaseMRM2(Module):
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

    def forward(self, x,used_ttrm2 = False,alpha = 1,beta = 1):
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
        x = self.fc_dropout(x)
        tcrm2_out = self.fc(x)
        ttrm2_out = None

        if used_ttrm2:
            tcrm2_softmax_out = torch.softmax(tcrm2_out, dim=1) ** alpha
            similar_x_x = torch.matmul(x, x.T)
            similar_x_x_stand = similar_x_x * beta / (similar_x_x.std(1, keepdim=True)+1e-8)
            ttrm2_out = torch.matmul((similar_x_x_stand),
                                     tcrm2_softmax_out / (tcrm2_softmax_out.sum(dim=0, keepdim=True)+1e-8))

        return {
            "tcrm2_out": tcrm2_out,
            "ttrm2_out": ttrm2_out,
            "x_embeddings": x
        }


class RNN_FCNMRM2(_RNN_FCN_BaseMRM2):
    _cell = nn.RNN


class LSTM_FCNMRM2(_RNN_FCN_BaseMRM2):
    _cell = nn.LSTM


class GRU_FCNMRM2(_RNN_FCN_BaseMRM2):
    _cell = nn.GRU


class MRNN_FCNMRM2(_RNN_FCN_BaseMRM2):
    _cell = nn.RNN

    def __init__(self, *args, se=16, **kwargs):
        super().__init__(*args, se=16, **kwargs)


class MLSTM_FCNMRM2(_RNN_FCN_BaseMRM2):
    _cell = nn.LSTM

    def __init__(self, *args, se=16, **kwargs):
        super().__init__(*args, se=16, **kwargs)


class MGRU_FCNMRM2(_RNN_FCN_BaseMRM2):
    _cell = nn.GRU

    def __init__(self, *args, se=16, **kwargs):
        super().__init__(*args, se=16, **kwargs)


if __name__ == '__main__':
    a = torch.ones((128, 1, 35))
    model = LSTM_FCNMRM2(1, 4,shuffle = False)
    out = model(a,torch.randint(0,2,(128,))*-1)
    for key in out:
        if out[key] is not None:
            print(out[key].shape)