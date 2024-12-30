import torch.nn as nn
import torch
import math
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np

class SemskeConv(nn.Module):
    
    def __init__(self, node_num=100, bias=True):
        super(SemskeConv, self).__init__()
        self.node_num = node_num

        A_sem = nn.Parameter(torch.zeros(node_num, node_num)) #Af
        self.A_sem = A_sem
        self.M = nn.Parameter(torch.zeros(node_num, node_num))
        self.W = nn.Parameter(torch.zeros(node_num, node_num))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(node_num))
            stdv = 1. / math.sqrt(self.M.size(1))
            self.bias.data.uniform_(-stdv, stdv)
            
        else:
            self.register_parameter('bias', None)

    def forward(self, input):

        self.A_sem = nn.Parameter(torch.where(torch.isnan(self.A_sem), torch.full_like(self.A_sem, 0), self.A_sem)).cuda()
        self.W = nn.Parameter(torch.where(torch.isnan(self.W), torch.full_like(self.W, 0), self.W))
        self.M = nn.Parameter(torch.where(torch.isnan(self.M), torch.full_like(self.M, 0), self.M))
        Adj = self.A_sem
        Adj_W = torch.mul(Adj, self.W)
        support = torch.matmul(input, Adj_W)
        output = torch.matmul(support, self.M)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

class _GraphConv(nn.Module):
    def __init__(self, in_features, hidden_feature, node_num, p_dropout= 0):
        super(_GraphConv, self).__init__()
        
        self.gconv1 = SemskeConv(node_num)
        self.bn = nn.BatchNorm1d(node_num * hidden_feature)

        self.gconv2 = SemskeConv(node_num)

        self.tanh = nn.Tanh()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        
        y = self.gconv1(x)

        y = self.tanh(y)
        if self.dropout is not None:
            y = self.dropout(y)
        y = self.gconv2(y)

        y = self.tanh(y)
        y = y + x

        return y


class Generator(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, node_num, batch_size, num_layers=2):
        super(Generator, self).__init__()


        self.hidden_prev = nn.Parameter(torch.zeros(num_layers, batch_size, hidden_size))
        

        self.GRU = nn.GRU(input_size=input_size,hidden_size=hidden_size,
                          num_layers=num_layers, dropout=0, batch_first = True)
                          
        
        self.GCN = _GraphConv(1, 10, node_num)
        
        self.linear = nn.Linear(hidden_size, output_size)


    def forward(self, x, hidden_size):

        # GCN block
        x = x.permute(0, 2, 1)
        GCN_set = self.GCN(x)
        
        
        x = GCN_set.reshape(x.shape[0],x.shape[1],x.shape[2])
        x = x.permute(0, 2, 1)
        

        out, h = self.GRU(x, self.hidden_prev)
        out = out.reshape(-1, hidden_size)
        out = self.linear(out)
        out = out.unsqueeze(dim=0)
        return out, h
    
class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, output_feature, p_dropout, num_stage=1, node_n=48, no_mapping=False):
        """

        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        if no_mapping:
            self.gc_out = GraphConvolution(hidden_feature, output_feature, node_n=node_n)
        else:
            self.gc_out = nn.Sequential(
                nn.Linear(hidden_feature, hidden_feature),
                nn.ReLU(),
                nn.Linear(hidden_feature, output_feature)
            )

        self.do = nn.Dropout(p_dropout)

        self.act_f = nn.Tanh()

    def forward(self, x):
        # print(x[0][0].cpu().detach().numpy())
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        y = self.gc_out(y)
        # print(y[0][0].cpu().detach().numpy())
        # y = y + x

        return y
    
class Channel_GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, output_feature, p_dropout, num_stage=1, node_n=48, channel=3, no_mapping=False):
        super().__init__()
        self.channel = channel
        self.output_feature = output_feature
        # self.list = []
        # for _ in range(channel):
        #     self.list.append(GCN(input_feature, hidden_feature, output_feature, p_dropout, num_stage, node_n))
        self.GCN = GCN(input_feature, hidden_feature, output_feature, p_dropout, num_stage, node_n*channel, no_mapping)
        
        # self.list = nn.ModuleList(self.list)
        
    def forward(self, x):
        '''
        x: (B, C, nodes, input_feature)
        '''
        B, C, N, _ = x.shape
        return self.GCN(x.reshape(x.shape[0], -1, x.shape[-1])).reshape((B, C, N, self.output_feature))
        # output = []
        # for i in range(self.channel):
        #     output.append(self.list[i](x[:, i]))
        # return torch.stack(output, dim=1)
    
class GCN_xyzr(nn.Module):
    def __init__(self, input_feature, hidden_feature, output_feature, p_dropout, num_stage=1, node_n=48, no_mapping=False):
        super().__init__()
        self.output_feature = output_feature
        self.GCN_xyz = Channel_GCN(input_feature, hidden_feature, output_feature, p_dropout, num_stage, node_n, 3, no_mapping)
        self.GCN_r = Channel_GCN(input_feature, hidden_feature, output_feature, p_dropout, num_stage, node_n, 4, no_mapping)
        
    def forward(self, x, r):
        '''
        x: (B, 3, nodes, input_feature)
        r: (B, 4, nodes, input_feature)
        '''
        x_out = self.GCN_xyz(x)
        r_out = F.normalize(self.GCN_r(r), dim=1)

        return x_out, r_out
    
def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

