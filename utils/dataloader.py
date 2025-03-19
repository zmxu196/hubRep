import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
import scipy.sparse as sp

class ThreeSources(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path)

        self.feature = data['data'][0]
        for i in range(self.feature.shape[0]):
            self.feature[i] = self.feature[i].T
        self.x1 = self.feature[0].astype(np.float32)
        self.x2 = self.feature[1].astype(np.float32)        
        self.x3 = self.feature[2].astype(np.float32)        
        self.label = data['truelabel'][0][0].squeeze()
        self.feature_list = [self.x1,self.x2,self.x3]

        for i in range(len(self.feature_list)): 
            self.__load__(self.feature_list[i], self.label, i)

    def __load__(self, feature, label, i):
        features = sp.csr_matrix(feature, dtype=np.float32)
        labels = _encode_onehot(label.reshape(-1))#.astype(np.uint8)
        self.num_labels = labels.shape[1]
        self.feature_list[i] = np.asarray(features.todense())
        self.label = np.where(labels)[1]


class BBCSport(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path)
        self.data1 = sp.csr_matrix(data['X'][0, 0])
        self.data2 = sp.csr_matrix(data['X'][0, 1])
        self.label = data['Y'].flatten()
        self.feature_list = [self.data1,self.data2]
        for i in range(len(self.feature_list)): 
            self.__load__(self.feature_list[i], self.label, i)

    def __load__(self, feature, label, i):
        features = sp.csr_matrix(feature, dtype=np.float32)
        labels = _encode_onehot(label.reshape(-1))#.astype(np.uint8)
        self.num_labels = labels.shape[1]
        self.feature_list[i] = np.asarray(features.todense())
        self.label = np.where(labels)[1]


class BDGP(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()
        self.data1 = data1
        self.data2 = data2
        self.label = labels
        self.feature_list = [self.data1,self.data2]

        for i in range(len(self.feature_list)):
            self.__load__(self.feature_list[i], self.label, i)

    def __load__(self, feature, label,i):
        features = sp.csr_matrix(feature, dtype=np.float32)
        labels = _encode_onehot(label.reshape(-1))#.astype(np.uint8)
        self.num_labels = labels.shape[1]
        self.feature_list[i] = np.asarray(features.todense())
        self.label = np.where(labels)[1]


def _encode_onehot(labels):
    classes = list(sorted(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.asarray(list(map(classes_dict.get, labels)),
                               dtype=np.int32)
    return labels_onehot

def load_data(dataset):

    path = './dataset/'

    if dataset == "3Sources":
        dataset = ThreeSources(path+'/3sources.mat')
        dims = [3560, 3631,3068]
        view = 3
        data_size = 169
        class_num = 6
    elif dataset == "BBCSport":
        dataset = BBCSport(path+'/BBCSport.mat')
        dims = [3183, 3203]
        view = 2
        data_size = 544
        class_num = 5

    elif dataset == "BDGP":
        dataset = BDGP(path+'/')
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5

    else:
        raise NotImplementedError

    return dataset, dims, view, data_size, class_num


def get_data(data):
    num_view = data[2]
    assert len(data[0].feature_list) == num_view
    feature_list = []
    for i in range(num_view):
        feature_list.append(torch.from_numpy(data[0].feature_list[i]))
        assert data[0].feature_list[i].shape[0] == data[-2]
    y = data[0].label
    n_clusters = len(np.unique(y))
    assert n_clusters == data[-1]
    feat_dims = data[1]

    return feature_list,num_view,feat_dims, y, n_clusters