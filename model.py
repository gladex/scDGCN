import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
#from torch_geometric.nn import GATConv

class GATcovLayer(nn .Module):
    def __init__(self,in_features,out_features,dropout=0.2,alpha=True,concat=True):
        super(GATcovLayer,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha 
        self.w = nn.Parameter(torch.zeros(size=(in_features,out_features)))
        nn.init.xavier_uniform_(self.w.data,gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features,1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.614)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input_h, adj):
        h = torch.mm(input_h, self.w)
        N = h.size()[0]
        input_concat = torch.cat([h.repeat(1,N).view(N * N,-1),h.repeat(N,1)],dim=1).\
            view(N,-1,2 * self.out_features)

        e = self.leakyrelu(torch.matmul(input_concat,self.a).squeeze(2))
        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0,e,zero_vec)
        attention = F.softmax(attention,dim=1) 
        attention = F.dropout(attention,self.dropout,training=self.training) 
        output_h = torch.matmul(attention,h)
        return output_h
                                        





class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout=0.6, alpha=True, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):

        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters() 
 
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight) 

        
        if self.use_bias:
            init.zeros_(self.bias)
 
    def forward(self, input_feature, adjacency):

        support = torch.mm(input_feature, self.weight)
        #support=support.to(torch.float64)
        adjacency=adjacency.to(torch.float32)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output
 
    """ def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.input_dim) + ' -> ' \
            + str(self.output_dim) + ')' """

class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        # encoder
        self.enc_1 = Linear(n_input, n_enc_1)
        self.BN1 = nn.BatchNorm1d(n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.BN2 = nn.BatchNorm1d(n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.BN3 = nn.BatchNorm1d(n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        # decoder
        self.dec_1 = Linear(n_z, n_dec_1)
        self.BN4 = nn.BatchNorm1d(n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.BN5 = nn.BatchNorm1d(n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.BN6 = nn.BatchNorm1d(n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    # z is the hidden layer, x_bar is the reconstruction layer
    def forward(self, x):
        enc_h1 = F.relu(self.BN1(self.enc_1(x)))
        enc_h2 = F.relu(self.BN2(self.enc_2(enc_h1)))
        enc_h3 = F.relu(self.BN3(self.enc_3(enc_h2)))

        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.BN4(self.dec_1(z)))
        dec_h2 = F.relu(self.BN5(self.dec_2(dec_h1)))
        dec_h3 = F.relu(self.BN6(self.dec_3(dec_h2)))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z, dec_h3


class SDCN(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1):
        super(SDCN, self).__init__()
        # AE to obtain internal information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,

            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,

            n_input=n_input,
            n_z=n_z).cuda()

        self.ae.load_state_dict(torch.load('model/Zeisel.pkl', map_location='cpu'))
        # GCN for inter information
        self.gnn_1 = GraphConvolution(n_input, n_enc_1)
        self.gnn_2 = GraphConvolution(n_enc_1, n_enc_2)
        self.gnn_3 = GraphConvolution(n_enc_2, n_enc_3)
        self.gnn_4 = GraphConvolution(n_enc_3, n_z)
        self.gnn_5 = GraphConvolution(n_z, n_clusters)
        #GATConv
        """ self.gnn_1 = GraphAttentionLayer(n_input, n_enc_1)
        self.gnn_2 = GraphAttentionLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GraphAttentionLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GraphAttentionLayer(n_enc_3, n_z)
        self.gnn_5 = GraphAttentionLayer(n_z, n_clusters) """

        #GATLayer
        """ self.gnn_1 = GATcovLayer(n_input, n_enc_1)
        self.gnn_2 = GATcovLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GATcovLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GATcovLayer(n_enc_3, n_z)
        self.gnn_5 = GATcovLayer(n_z, n_clusters) """
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))

        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        
        # Fill the input "Tensor" with values according to the method
        # and the resulting tensor will have the values sampled from it
        self._dec_mean = nn.Sequential(nn.Linear(n_dec_3, n_input), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(n_dec_3, n_input), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(n_dec_3, n_input), nn.Sigmoid())

        # degree
        self.v = v
        self.zinb_loss = ZINBLoss().cuda()

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z, dec_h3 = self.ae(x)
        sigma = 0.5
        # GCN Module
        h = self.gnn_1(x, adj)


        h = self.gnn_2((1 - sigma) * h + sigma * tra1, adj)

        h = self.gnn_3((1 - sigma) * h + sigma * tra2, adj)

        h = self.gnn_4((1 - sigma) * h + sigma * tra3, adj)

        h = self.gnn_5((1 - sigma) * h + sigma * z, adj)
        # The last layer (multiple classification layer with softmax function)
        predict = F.softmax(h, dim=1)
        # Dual Self-supervised Module
        _mean = self._dec_mean(dec_h3)
        _disp = self._dec_disp(dec_h3)
        _pi = self._dec_pi(dec_h3)
        zinb_loss = self.zinb_loss
        z = z.cpu()
        h = h.cpu()
        # qij
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)

        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        q = q.cuda()
        return x_bar, q, predict, h, _mean, _disp, _pi, zinb_loss
    




class ZINBLoss(nn.Module):
    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0,    ridge_lambda=0.0):
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor
        
        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0-pi+eps)
        zero_nb = torch.pow(disp/(disp+mean+eps), disp)
        zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)
        
        if ridge_lambda > 0:
            ridge = ridge_lambda*torch.square(pi)
            result += ridge
        result = torch.mean(result)
        return result

class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()
    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)

class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()
    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)