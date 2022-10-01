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


class OS_CNNMRM2(nn.Module):
    def __init__(self,layer_parameter_list,n_class,few_shot = True):
        super(OS_CNNMRM2, self).__init__()
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

    def forward(self, x, used_ttrm2 = False,alpha = 1,beta = 1):

        x = self.net(x)

        # X = self.averagepool(X)
        # X = X.squeeze_(-1)

        x = self.global_average_pool(x).squeeze(-1)
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

