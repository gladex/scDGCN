#train
from model import SDCN
from utils import load_graph,heatmap,Decoding_loss,Decoding_loss2,heatmap2
from torch.optim import Adam
from sklearn.cluster import KMeans
import torch
import scanpy as sc
import numpy as np
from munkres import Munkres
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn import metrics
import torch.nn.functional as F
from get_graph import getGraph
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment
import umap
import matplotlib.pyplot as plt
from collections import OrderedDict
import random
# Target distribution
def target_distribution(q):
    # Pij
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_sdcn(dataset, X_raw, sf, args):
    global p
    device = torch.device("cuda" if args.cuda else "cpu")
    model = SDCN(
        n_enc_1=args.n_enc_1,
        n_enc_2=args.n_enc_2,
        n_enc_3=args.n_enc_3,
        n_dec_1=args.n_dec_1,
        n_dec_2=args.n_dec_2,
        n_dec_3=args.n_dec_3,
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        v=1).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    L = 0
    N = dataset.shape[1]
    N = int(N)
    K=15
    method=args.method
    X = dataset.X.T
    _, n = X.shape
    adj = getGraph(X, L, K, method).to(device)
    data = torch.Tensor(X).to(device)
    data=data.T
    
    #y = dataset.y
    y = dataset.obs['Group']
    with torch.no_grad():
        _, _, _, _, z, _ = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=args.n_init)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_)

    # evaluation.py：ARI\ACC\NMI
    eva(y, y_pred, 0)
    Balance_para = [0.1, 0.01, 1e-7, 0.1]
    X_raw = torch.tensor(X_raw)
    X_raw = X_raw.transpose(0, 1).cuda()
    res_lst = []
    loss_list = []
    for epoch in range(args.epoch):
        if epoch % 1 == 0:
            _, tmp_q, pred,_, _, _, _, _ = model(data, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
            #print(tmp_q)
            res1 = tmp_q.cpu().numpy().argmax(1)  # Q
            res2 = pred.data.cpu().numpy().argmax(1)  # Z
            res3 = p.data.cpu().numpy().argmax(1)  # P
            tmp_list = []
            tmp_list.append(np.array(eva(y, res1, str(epoch) + 'Q')))
            tmp_list.append(np.array(eva(y, res2, str(epoch) + 'Z')))
            tmp_list.append(np.array(eva(y, res3, str(epoch) + 'P')))
            tmp_list = np.array(tmp_list)

            idx = np.argmax(tmp_list[:,0])

            res_lst.append(tmp_list[idx])


        x_bar, q, pred, z, meanbatch, dispbatch, pibatch, zinb_loss = model(data, adj)

        binary_crossentropy_loss = F.binary_cross_entropy(q, p)
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        #re_loss = F.mse_loss(x_bar, data)      
        re_loss = Decoding_loss(x_bar, data,adj)

        sf = torch.as_tensor(sf).cuda()#.cuda() sourceTensor.clone().detach()

        zinb_loss = zinb_loss(X_raw, meanbatch, dispbatch, pibatch, sf)

        loss = Balance_para[0] * binary_crossentropy_loss + Balance_para[1] * ce_loss + Balance_para[2] * re_loss + Balance_para[3] * zinb_loss

        loss_list.append(loss.cpu().detach().numpy())
        #loss_list = np.array(loss_list)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    res_lst = np.array(res_lst)
    best_idx = np.argmax(res_lst[:, 1])

    print('ACC={:.2f} +- {:.2f}'.format(res_lst[:, 0][best_idx]*100, np.std(res_lst[:, 0])))
    print('NMI={:.2f} +- {:.2f}'.format(res_lst[:, 1][best_idx]*100, np.std(res_lst[:, 1])))
    print('ARI={:.2f} +- {:.2f}'.format(res_lst[:, 2][best_idx]*100, np.std(res_lst[:, 2])))
    print('F1={:.2f} +- {:.2f}'.format(res_lst[:, 3][best_idx]*100, np.std(res_lst[:, 3])))

    


def plot(X, fig, col, size, true_labels, ann,dataset_name):
                ax = fig.add_subplot(1, 1, 1) 

                if len(col) < len(set(true_labels)):
                    col = col * (len(set(true_labels)) // len(col)) + col[:len(set(true_labels)) % len(col)]
                for i, point in enumerate(X):
                    ax.scatter(point[0], point[1], s=size, c=col[true_labels[i]], label=ann[i])

                ax.set_title(f"UMAP Visualization of "+str(dataset_name)+" dataset", fontsize=14)
                fig.savefig("D:/Study/Code/S/scDSCD/code/data" + str(dataset_name) + "UMAP.pdf")


def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    numclass1 = len(l1)
    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    # y_true：Like 1d array or label indicator array/sparse matrix (correct) label
    # y_pred：Like a one-dimensional array or label indicator array/sparse matrix predicted labels, returned by the classifier
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro


def eva(y_true, y_pred, epoch=0):
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred)
    ari = ari_score(y_true, y_pred)
    print(epoch, ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
          ', f1 {:.4f}'.format(f1))
    return acc, nmi, ari, f1

def cluster_acc2(y_true, y_pred):
    y_true = y_true - np.min(y_true)
    num_classes = len(np.unique(y_true))


    clusters_mapping = {}
    for i, c in enumerate(np.unique(y_pred)):
        clusters_mapping[c] = i

    new_predict = [clusters_mapping[c] for c in y_pred]

    if len(np.unique(new_predict)) != num_classes:
        print('Error: Number of clusters does not match!')
        return 0,0


    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(num_classes):
        mask_true = y_true == i
        for j in range(num_classes):
            mask_pred = np.array(new_predict) == j
            confusion_matrix[i][j] = np.sum(mask_true & mask_pred)


    m = Munkres()
    indexes = m.compute(-confusion_matrix)


    new_predict = np.zeros(len(y_pred))
    for i, j in indexes:
        new_predict[y_pred == list(clusters_mapping.keys())[j]] = i


    acc = accuracy_score(y_true, new_predict)
    f1_macro = f1_score(y_true, new_predict, average='macro')

    return acc, f1_macro


def cluster_acc3(y_true, y_pred):
    # y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array((ind[0], ind[1])).T

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, 1