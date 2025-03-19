"""
Hubness metrics. Adapted from https://scikit-hubness.readthedocs.io/en/latest/
"""
import torch as th

def npy(t, to_cpu=True):
    """
    Convert a tensor to a numpy array.

    :param t: Input tensor
    :type t: th.Tensor
    :param to_cpu: Call the .cpu() method on `t`?
    :type to_cpu: bool
    :return: Numpy array
    :rtype: np.ndarray
    """
    if isinstance(t, (list, tuple)):
        # We got a list. Convert each element to numpy
        return [npy(ti) for ti in t]
    elif isinstance(t, dict):
        # We got a dict. Convert each value to numpy
        return {k: npy(v) for k, v in t.items()}
    # Assuming t is a tensor.
    if to_cpu:
        return t.cpu().detach().numpy()
    return t.detach().numpy()


def skewness(k_occ):
    centered = k_occ - k_occ.mean(dim=-1, keepdims=True)
    m3 = (centered**3).mean(dim=-1)
    m2 = (centered**2).mean(dim=-1)
    skew = m3 / (m2 ** (3/2))
    return npy(skew)


def hub_occurrence(k_occ, k, hub_size=2):
    is_hub = (k_occ >= (hub_size * k)).float()
    hub_occ = (k_occ * is_hub).mean(dim=-1) / k
    return npy(hub_occ)


def antihub_occurrence(k_occ):
    is_antihub = (k_occ == 0).float()
    return npy(is_antihub.mean(dim=-1))


def robinhood_index(k_occ):
    numerator = .5 * th.sum(th.abs(k_occ - k_occ.mean(dim=-1, keepdims=True)), dim=-1)
    denominator = th.sum(k_occ, dim=-1)
    return npy(numerator / denominator)



def good_bad_hub_occurrence(k_occ,indices,GT_labels,k,threshold):
    good_hubs = []
    bad_hubs = []

    hub_size=2
    # is_hub = (k_occ >= (hub_size * k)).float()

    n_samples = len(GT_labels)
    good_hubs = []
    bad_hubs = []

    good_hub_occ = []
    bad_hub_occ = []

    isGoodHub = th.zeros(n_samples, dtype=th.float32)  # 全 0 张量，表示 False
    isBadHub = th.zeros(n_samples, dtype=th.float32) 

    is_hub = (k_occ >= (hub_size * k))  # Identify hubs based on k_occ
    hub_indices = th.nonzero(is_hub).squeeze()  # 只对 hub 样本进行操作
    if hub_indices.dim() > 0: 
        consistency_ratio_list = []
        good_consistency_ratio_list = []
        bad_consistency_ratio_list = []

        for i in hub_indices:
            neighbor_labels = GT_labels[indices[i]]  # 取出最近邻的标签
            sample_label = GT_labels[i]  # 样本自身的标签

            # 计算 consistency ratio (邻居中与样本标签一致的比例)
            consistency_ratio = (neighbor_labels == sample_label).sum().item() / len(neighbor_labels)
            consistency_ratio_list.append(consistency_ratio)
            # 判断是 good hub 还是 bad hub
            if consistency_ratio >= threshold:
                good_hubs.append(i.item())  # Good Hub
                # good_hub_occ.append(k_occ[i].item())
                isGoodHub[i] = 1.0
                good_consistency_ratio_list.append(consistency_ratio)
            else:   
                bad_hubs.append(i.item())  # Bad Hub
                # bad_hub_occ.append(k_occ[i].item())
                isBadHub[i] = 1.0
                bad_consistency_ratio_list.append(consistency_ratio)
        # good_hub_occ2 = numpy.array(good_hub_occ).sum() / n_samples / k # same

        if len(consistency_ratio_list) > 0:
            consistency_ratio_avg = sum(consistency_ratio_list) / len(consistency_ratio_list)
        else:
            consistency_ratio_avg = 0  # 或者设为 None，取决于你想如何处理

        if len(good_consistency_ratio_list) > 0:
            good_consistency_ratio_avg = sum(good_consistency_ratio_list) / len(good_consistency_ratio_list)
            good_hub_occ = (k_occ.cpu() * isGoodHub).mean(dim=-1) / k  
        else:
            good_consistency_ratio_avg = 0  # 或者设为 None，取决于你想如何处理
            good_hub_occ = 0 

        # 计算 bad consistency 的平均值，避免空列表时除零错误
        if len(bad_consistency_ratio_list) > 0:
            bad_consistency_ratio_avg = sum(bad_consistency_ratio_list) / len(bad_consistency_ratio_list)
            bad_hub_occ = (k_occ.cpu() * isBadHub).mean(dim=-1) / k 
        else:
            bad_consistency_ratio_avg = 0  # 或者设为 None，取决于你想如何处理
            bad_hub_occ = 0

        good_hub_ratio = isGoodHub.sum().item() / n_samples    # 好中心点的比例    
        bad_hub_ratio =  isBadHub.sum().item()  / n_samples    # 坏中心点的比例
        hub_ratio =      is_hub.sum().item()    / n_samples    # 中心点的比例

        if is_hub.sum().item() > 0:
            # good_hub_proportion = isGoodHub.sum().item() / is_hub.sum().item()  # 好中心点的比例
            # bad_hub_proportion = isBadHub.sum().item() / is_hub.sum().item()    # 坏中心点的比例
            good_hub_proportion = isGoodHub.sum().item()    # 好中心点的比例
            bad_hub_proportion = isBadHub.sum().item()    # 坏中心点的比例
        else:
            good_hub_proportion = 0
            bad_hub_proportion = 0
        assert isGoodHub.sum().item() + isBadHub.sum().item() == is_hub.sum().item() 

    else:
        good_hub_occ = 0 
        bad_hub_occ = 0
        hub_ratio = 0
        good_hub_ratio = 0
        bad_hub_ratio = 0
        good_hub_proportion = 0
        bad_hub_proportion = 0
        consistency_ratio_avg = 0
        good_consistency_ratio_avg = 0
        bad_consistency_ratio_avg = 0

    return good_hub_occ,bad_hub_occ,hub_ratio,good_hub_ratio,bad_hub_ratio,good_hub_proportion,bad_hub_proportion,consistency_ratio_avg,good_consistency_ratio_avg,bad_consistency_ratio_avg

def score(k_occ,indices, GT_labels,k,threshold):
    k_occ = k_occ.float()
    goodHO, badHO,HubRatio,goodHubRatio,badHubRatio,goodHubProp,badHubProp,consistency_ratio_avg,good_consistency_ratio_avg,bad_consistency_ratio_avg = good_bad_hub_occurrence(k_occ,indices,GT_labels,k,threshold)

    return {
        "SK": skewness(k_occ),
        "HO": hub_occurrence(k_occ, k),
        "goodHO":goodHO,
        "badHO": badHO,
        "HubRatio": HubRatio,
        "goodHubRatio":goodHubRatio,
        "badHubRatio": badHubRatio,
        "goodHubNumber":goodHubProp,
        "badHubNumber": badHubProp,
        "consistency_ratio_avg": consistency_ratio_avg,    
        "good_consistency_ratio_avg": good_consistency_ratio_avg,    
        "bad_consistency_ratio_avg": bad_consistency_ratio_avg,     
        "antiHO": antihub_occurrence(k_occ),
        "robin": robinhood_index(k_occ),
    }

