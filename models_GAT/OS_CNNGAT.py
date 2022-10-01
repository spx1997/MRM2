import torch
import torch.nn as nn
import torch.nn.functional as F



class SampaddingConv1D_BN(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size):
        super(SampaddingConv1D_BN, self).__init__()
        self.padding = nn.ConstantPad1d((int((kernel_size-1)/2), int(kernel_size/2)), 0)
        self.conv1d = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.bn = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, X):
        X = self.padding(X)
        X = self.conv1d(X)
        X = self.bn(X)
        return X

class build_layer_with_layer_parameter(nn.Module):
    def __init__(self,layer_parameters):
        super(build_layer_with_layer_parameter, self).__init__()
        self.conv_list = nn.ModuleList()

        for i in layer_parameters:
            conv = SampaddingConv1D_BN(i[0],i[1],i[2])
            self.conv_list.append(conv)

    def forward(self, X):

        conv_result_list = []
        for conv in self.conv_list:
            conv_result = conv(X)
            conv_result_list.append(conv_result)

        result = F.relu(torch.cat(tuple(conv_result_list), 1))
        return result


class OS_CNNGAT(nn.Module):
    def __init__(self,layer_parameter_list,n_class,few_shot = True):
        super().__init__()
        self.few_shot = few_shot
        self.layer_parameter_list = layer_parameter_list
        self.layer_list = []


        for i in range(len(layer_parameter_list)):
            layer = build_layer_with_layer_parameter(layer_parameter_list[i])
            self.layer_list.append(layer)

        self.net = nn.Sequential(*self.layer_list)

        self.global_average_pool = nn.AdaptiveAvgPool1d(1)

        out_put_channel_numebr = 0
        for final_layer_parameters in layer_parameter_list[-1]:
            out_put_channel_numebr = out_put_channel_numebr+ final_layer_parameters[1]

        self.fc = nn.Linear(out_put_channel_numebr, n_class)

        self.transformation = nn.Linear(out_put_channel_numebr,out_put_channel_numebr)
        self.a = nn.Parameter(torch.randn(out_put_channel_numebr * 2, 1))
        self.a.requires_grad = True
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

        self.gnn_fc_out = nn.Sequential(
                                nn.Linear(out_put_channel_numebr,out_put_channel_numebr),
                                nn.LeakyReLU(),
                                nn.Linear(out_put_channel_numebr,n_class),
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

    def forward(self, x, flags = None,alpha = 1,beta = 1):

        x = self.net(x)

        # X = self.averagepool(X)
        # X = X.squeeze_(-1)

        hidden = self.global_average_pool(x).squeeze(-1)
        origin_out = self.fc(hidden)

        att_weight = self.cal_attention(hidden)
        hidden = att_weight.mm(hidden) + hidden
        gat_out = self.gnn_fc_out(hidden)
        return {
            "origin_out": origin_out,
            "gat_out": gat_out,
            "x_embeddings": hidden
        }