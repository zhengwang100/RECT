import sys
import random
import numpy as np
from numpy import linalg as la
import scipy.sparse as sp
import torch
import torch.nn as nn
np.set_printoptions(threshold=sys.maxsize)
from sklearn.preprocessing import normalize

# MF_DEEPWALK DeepWalk as Matrix Factorization
def graph_of_deepwalk(G):
    '''Calculate: G = (G+G^2)/2
       Inputs: G - np[n,n], note the in the MFDW sparse matrix is not supported
    '''
    G = normalize(G, norm='l1', axis=1)
    G = (G + np.dot(G, G))/2
    return G

def read_features(feature_file):
    '''
        Note: only support one-hot features     
    '''
    #idx_features = np.genfromtxt(feature_file, dtype=np.int32)
    idx_features = np.genfromtxt(feature_file, dtype=np.float32)
    features = sp.csr_matrix(idx_features[:, 1:], dtype=np.float32)
    features = features.todense()
    return features

def svd_feature(features, d=200):
    if( features.shape[1] <= d ): return features       
    U, S, VT = la.svd(features)
    Ud = U[:, 0:d]
    Sd = S[0:d]
    features = np.array(Ud)*Sd.reshape(d)
    return features

def symmetrize(a):
    ##return a + a.T - sp.diags(a.diagonal())
    #return a + a.T - np.diag(a.diagonal())
    a = a + a.T.multiply(a.T > a) - a.multiply(a.T > a)
    return a.todense()
    
def read_graph_as_matrix(nodeids, edge_file):
    ''' Read a symmetric adjacency matrix from a file 
        Input: nodeids: [1,2,3,4,...]
        Return: the sparse adjacency matrix
    '''
    idx = np.array(nodeids, dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(edge_file, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(len(idx), len(idx)),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = symmetrize(adj)
    #print('symmetrice adj type', type(adj))
    return adj

###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    #print(rowsum.T)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    #print(rowsum.T)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from DeepRelax.basic_functions import get_data_file
    edge_file, label_file, feature_file = get_data_file()
    features = read_features(feature_file)
