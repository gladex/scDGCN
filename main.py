from __future__ import print_function, division
import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
from utils import load_data, load_graph
#from evaluation import eva
from torch.utils.data import DataLoader, TensorDataset
import h5py
import scanpy as sc
from preprocess2 import read_dataset, normalize,normalize2
from train import train_sdcn
import pandas as pd
from readdata import prepro 
import time
#from layers import ZINBLoss, MeanAct, DispAct
#from GNN import GNNLayer
seed = 0
torch.manual_seed(seed) 
torch.cuda.manual_seed(seed) 
torch.cuda.manual_seed_all(seed) 

if __name__ == "__main__":
    # File = [gene_expresion data file]
    File = ['Zeisel']

    # model_para = [n_enc_1(n_dec_3), n_enc_2(n_dec_2), n_enc_3(n_dec_1), n_cluster, n_init]
    model_para = [512, 256, 64]

    # Para = [batch_size, lr, epoch]
    Para = [1024, 1e-4, 30]

    Fileformat = ['data.h5','csv','dataset.h5']

    # Cluster_para = [n_cluster, n_z, n_input, n_init]
    Cluster_para = [8, 32, 2000, 20]

    Method = ['pearson','spearman','NE']
    
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--name', type=str, default=File[0])
    parser.add_argument('--n_enc_1', default=model_para[0], type=int)
    parser.add_argument('--n_enc_2', default=model_para[1], type=int)
    parser.add_argument('--n_enc_3', default=model_para[2], type=int)
    parser.add_argument('--n_dec_1', default=model_para[2], type=int)
    parser.add_argument('--n_dec_2', default=model_para[1], type=int)
    parser.add_argument('--n_dec_3', default=model_para[0], type=int)
    parser.add_argument('--method', default=Method[2], type=str)
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--lr', type=float, default=Para[1])
    parser.add_argument('--epoch', type=int, default=Para[2])
    parser.add_argument('--n_clusters', default=Cluster_para[0], type=int)
    parser.add_argument('--n_z', default=Cluster_para[1], type=int)
    parser.add_argument('--n_input', type=int, default=Cluster_para[2])
    parser.add_argument('--n_init', type=int, default=Cluster_para[3])
    parser.add_argument('--format', type=str, default=Fileformat[2])
    # 解析参数
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))

    device = torch.device("cuda" if args.cuda else "cpu")

    start = time.time()
    #dataset = load_data(args.name)
    if args.format == 'data.h5':
        X,Y,var = prepro(args.name)
        X = np.ceil(X).astype(int)
        count_X = X
        adata = sc.AnnData(X,dtype='float32')
        adata.var_names = var.index
        adata.obs['Group'] = Y
    elif args.format == 'csv':
        adata=sc.read_csv('data/PBMC/pbmc.csv')
        #adata=adata.T#MTAB需要
        y= pd.read_csv('data/PBMC/clusters.csv', index_col=0)
        y= np.array(y.values)
        adata.obs['Group'] = y
        adata.var_names_make_unique()
    elif args.format == 'dataset.h5':
        data_mat = h5py.File('data/'+args.name+'.h5')
        x = np.array(data_mat['X'])
        #x=x.T#klein.h5,romanov.h5需要
        y = np.array(data_mat['Y'])
        data_mat.close()
        #print(x.shape)
        #print(y.shape)

# preprocessing scRNA-seq read counts matrix
        adata = sc.AnnData(x,dtype='float32')
        adata.obs['Group'] = y
    else:
        print(f'Unknown name: {args.name}')



    adata = read_dataset(adata,
                         transpose=False,
                         test_split=False,
                         copy=True)
    adata = normalize2(adata,
                size_factors=True,
                normalize_input=True,
                logtrans_input=True,
                select_hvg=True)
    """ adata = normalize(adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True) """



    X = adata.X
    X=X.T
    X_raw = adata.raw.X.T
    sf = adata.obs.size_factors
    train_sdcn(adata, X_raw, sf, args)
    time_used = time.time()-start
    print(time_used)
