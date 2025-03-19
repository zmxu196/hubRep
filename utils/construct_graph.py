import os
import pdb
import time
import torch
import openpyxl
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from utils.hubness.get_hub import get_hub_score,get_hub_occOnly
from .ops_al import ZScoreNorm,CL2N
import scipy.io
import scipy.sparse as sp
import faiss



def load_single_view_data(logger, feature, k_nearest_neighbors, prunning_one):
    hub_occ = get_hub_occOnly(feature,k_nearest_neighbors,dist_metric="sqeuclidean") 
    adj, adj_wave, adj_hat = construct_adjacency_matrix(logger, feature.detach().cpu(), k_nearest_neighbors, prunning_one)
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    weight_mask = adj_wave.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight

    return adj_wave.cuda(),adj_hat.cuda(),hub_occ,norm,weight_tensor


def construct_adjacency_matrix(logger, features, k_nearest_neighbors, prunning_one):
    start_time = time.time()
    features = features.cpu().numpy().astype('float32')
    index = faiss.IndexFlatL2(features.shape[1]) 
    index.add(features)
    D, I = index.search(features, k_nearest_neighbors + 1) 
    row = np.repeat(np.arange(features.shape[0]), k_nearest_neighbors)
    col = I[:, 1:].flatten()
    data = np.ones_like(col, dtype=np.float32)

    adj_wave = torch.sparse_coo_tensor(
        indices=torch.LongTensor([row, col]),
        values=torch.FloatTensor(data),
        size=(features.shape[0], features.shape[0]),
        dtype=torch.float32
    )
    
    if prunning_one:
        adj_wave_dense = adj_wave.to_dense()
        symmetric_mask = (adj_wave_dense == adj_wave_dense.T)
        adj_wave_dense = adj_wave_dense * symmetric_mask
        adj_wave = adj_wave_dense.to_sparse()

    non_diag_mask = row != col
    row = row[non_diag_mask]
    col = col[non_diag_mask]
    data = data[non_diag_mask]

    adj = torch.sparse_coo_tensor(
        indices=torch.LongTensor([row, col]),
        values=torch.FloatTensor(data),
        size=(features.shape[0], features.shape[0]),
        dtype=torch.float32
    )

    adj_hat = construct_adjacency_hat(adj)

    # logger.info(f"Time taken for construction: {time.time() - start_time:.2f} seconds")
    
    return adj, adj_wave, adj_hat


def construct_adjacency_hat(adj):
    """
    :param adj: Original adjacency matrix, type torch.sparse.Tensor
    :return: Normalized adjacency matrix, type torch.sparse.Tensor
    """
    adj_ = adj + torch.eye(adj.shape[0], device=adj.device).to_sparse()
    rowsum = torch.sparse.sum(adj_, dim=1).to_dense()
    degree_mat_inv_sqrt = torch.diag(torch.pow(rowsum, -0.5))
    adj_normalized = torch.matmul(degree_mat_inv_sqrt, torch.matmul(adj_.to_dense(), degree_mat_inv_sqrt))

    return adj_normalized.to_sparse()

def construct_sparse_float_tensor(np_matrix):
    """
    Construct a sparse float tensor from a numpy matrix
    :param np_matrix: <class 'numpy.ndarray'>
    :return: torch.sparse.FloatTensor
    """
    sp_matrix = sp.coo_matrix(np_matrix)
    coords = np.vstack((sp_matrix.row, sp_matrix.col)).transpose()
    values = sp_matrix.data
    shape = sp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(
        torch.LongTensor(coords.T),
        torch.FloatTensor(values),
        torch.Size(shape)
    )
    return sparse_tensor


