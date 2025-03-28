
import torch
import torch.nn as nn
from typing import Optional, List
from collections import OrderedDict
from cytoolz.itertoolz import sliding_window

from layers.graph_conv import GraphConvolution

import scipy.io
import scipy.sparse as sp
def build_layer_units(layer_type: str, dims: List[int], act_func: Optional[nn.Module]) -> nn.Module:
    """
        Construct a multi-layer network accroding to layer type, dimensions and activation function
        Tips: the activation function is not used in the final layer
    :param layer_type: the type of each layer, such as linear or gcn
    :param dims: the list of dimensions
    :param act_func: the type of activation function
    :return:
    """
    layer_list = []
    for input_dim, output_dim in sliding_window(2, dims[:-1]):
        layer_list.append(single_unit(layer_type, input_dim, output_dim, act_func))

    layer_list.append(single_unit(layer_type, dims[-2], dims[-1], None))

    return nn.Sequential(*layer_list)


def single_unit(layer_type: str, input_dim: int, output_dim: int, act_func: Optional[nn.Module]):
    """
        Construct each layer
    :param layer_type: the type of current layer
    :param input_dim: the input dimension
    :param output_dim: the output dimension
    :param act_func: the activation function
    :return:
    """
    unit = []
    if layer_type == 'linear':
        unit.append(('linear', nn.Linear(input_dim, output_dim)))
    elif layer_type == 'gcn':
        unit.append(('gcn', GraphConvolution(input_dim, output_dim)))
    else:
        print("Please input correct layer type!")
        exit()

    if act_func is not None:
        unit.append(('act', act_func))

    return nn.Sequential(OrderedDict(unit))


def dot_product_decode(Z):
    """
        predicting the reconstructed adjacent matrix
    :param Z: embedding feature
    :return: reconstructed adjacent matrix
    """
    A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
    return A_pred


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def ZScoreNorm(x):

    means = x.mean(dim=-1, keepdims=True)
    stds = x.std(dim=-1, keepdims=True)
    eps = 1e-8  
    normed = (x - means) / (stds + eps)
    return normed

def l2_normalize(data,dim):
    norm = torch.norm(data, p=2, dim=dim, keepdim=True)
    return data / (norm + 1e-8)  

def CL2N(features, dim=-1):

        center = features.mean(dim=dim, keepdim=True)
        centered = features - center
        normed = l2_normalize(centered,dim)

        return normed