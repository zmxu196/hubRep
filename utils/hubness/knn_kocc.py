import torch as th
from scipy.stats import mode as scipy_mode  
import numpy

import numpy as np
from scipy.spatial import distance

def block_cdist(features, block_size=1000):
    n_samples = features.shape[0]
    distance_matrix = th.zeros((n_samples, n_samples), dtype=th.float32, device=features.device)
    
    for i in range(0, n_samples, block_size):
        end_i = min(i + block_size, n_samples)
        for j in range(0, n_samples, block_size):
            end_j = min(j + block_size, n_samples)
            # 计算块内的距离
            distance_matrix[i:end_i, j:end_j] = th.cdist(features[i:end_i], features[j:end_j])
    
    return distance_matrix

def batched_knn(dist, dist_to_kth_neighbor, batch_size=1024):
    num_samples = dist.size(0)
    p = th.zeros_like(dist, dtype=th.bool).cuda()

    
    for i in range(0, num_samples, batch_size):
        end_i = min(i + batch_size, num_samples)
        for j in range(0, num_samples, batch_size):
            end_j = min(j + batch_size, num_samples)
            p[i:end_i, j:end_j] = dist[i:end_i, j:end_j] <= dist_to_kth_neighbor[i:end_i].unsqueeze(1)
            
    return p
 
def k_occurrence_ONLY(features, k=5, metric="sqeuclidean"):
    n_samples = features.size(0)
    if metric == "cosine":
         normed = th.nn.functional.normalize(features, dim=1, p=2)
         dist = - (normed @ normed.transpose(0, 1))
    elif metric == "sqeuclidean":
        if n_samples > 1000:
            dist = block_cdist(features, block_size=1000)
        else:
            dist = th.cdist(features, features, p=2) 
    else:
        raise RuntimeError(f"Unknown k_occurrence metric: '{metric}'.")

    temp = th.topk(dist, k=k+1, dim=-1, largest=False)
    dist_to_kth_neighbor = temp[0][:, -1] 
    if n_samples < 1000:
        p = (dist <= dist_to_kth_neighbor.unsqueeze(1))
    else:
        p = batched_knn(dist, dist_to_kth_neighbor, batch_size=1000)

    k_occ = p.sum(axis=0)

    return k_occ

