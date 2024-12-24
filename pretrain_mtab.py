import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import Linear
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from train import eva
import torch.nn.functional as F
import pandas as pd
import scanpy as sc
from readdata import prepro 
from preprocess2 import read_dataset, normalize,normalize2
import h5py

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


    def forward(self, x):
        enc_h1 = F.relu(self.BN1(self.enc_1(x)))
        enc_h2 = F.relu(self.BN2(self.enc_2(enc_h1)))
        enc_h3 = F.relu(self.BN3(self.enc_3(enc_h2)))

        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.BN4(self.dec_1(z)))
        dec_h2 = F.relu(self.BN5(self.dec_2(dec_h1)))
        dec_h3 = F.relu(self.BN6(self.dec_3(dec_h2)))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar,  z


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def pretrain_ae(model, dataset, y):
    train_loader = DataLoader(dataset, batch_size=Para[0], shuffle=True)
    print(model)
    # Adam
    optimizer = Adam(model.parameters(), lr=Para[1])
    for epoch in range(Para[2]):
        #for batch_idx, (x, _) in enumerate(train_loader):
        for batch_idx, x in enumerate(train_loader):
            x = x.cuda()
            x_bar, _ = model(x)

            x_bar = x_bar.cpu()
            x = x.cpu()
            loss = F.mse_loss(x_bar, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            x = torch.Tensor(dataset).cuda().float()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            print('{} loss: {}'.format(epoch, loss))
            kmeans = KMeans(n_clusters=Cluster_para[0], n_init=Cluster_para[1]).fit(z.data.cpu().numpy())
            eva(y, kmeans.labels_, epoch)
        # Generate a pre-trained model
        torch.save(model.state_dict(), File[0])

# File = [Pre-training file, gene_expresion data file, labels file]
File = ['model/Bach_80.pkl', 'data/MTAB/mtab.txt', 'data/MTAB/mtab_label.txt']
# Para = [batch_size, lr, epoch]
Para = [1024, 1e-3, 100]
# model_para = [n_enc_1(n_dec_3), n_enc_2(n_dec_2), n_enc_3(n_dec_1)]
model_para = [512, 256, 64]
# Cluster_para = [n_cluster, n_init, n_input, n_z]
Cluster_para = [8, 20, 2000, 32]

model = AE(
            n_enc_1=model_para[0], n_enc_2=model_para[1], n_enc_3=model_para[2],
            n_dec_1=model_para[2], n_dec_2=model_para[1], n_dec_3=model_para[0], n_input=Cluster_para[2], n_z=Cluster_para[3], ).cuda()


#x = np.loadtxt(File[1], dtype=float)


#data.h5
""" adata=sc.read_csv('data/GSE154763/GSE154763.csv')
#adata=adata.T
y= pd.read_csv('data/GSE154763/clusters.csv', index_col=0)
y= np.array(y.values)
adata.obs['Group'] = y
t = adata.obs['Group'] """

X,Y,var = prepro('Bach')
X = np.ceil(X).astype(np.int)
count_X = X

adata = sc.AnnData(X)
adata.obs['Group'] = Y



#csv
""" adata=sc.read_csv('data/PBMC/pbmc.csv')
    #adata=adata.T
y= pd.read_csv('data/PBMC/clusters.csv', index_col=0)
y= np.array(y.values)
adata.obs['Group'] = y """


#dataset.h5
""" data_mat = h5py.File('data/Human2.h5')
x = np.array(data_mat['X'])
#x=x.T#klein.h5 romanov.h5
y = np.array(data_mat['Y'])
data_mat.close()
        #print(x.shape)
        #print(y.shape)

# preprocessing scRNA-seq read counts matrix
adata = sc.AnnData(x)
adata.obs['Group'] = y
adata.var_names_make_unique() """

adata = read_dataset(adata,
                         transpose=False,
                         test_split=False,
                         copy=True)

adata = normalize2(adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)
y=adata.obs['Group']

dataset = adata.X
#dataset = LoadDataset(x)
pretrain_ae(model, dataset, y)
