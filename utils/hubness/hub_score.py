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
def hub_score(k_occ,k):
    k_occ = k_occ.float()

    return {
        "SK": skewness(k_occ),
        "HO": hub_occurrence(k_occ, k),
        "antiHO": antihub_occurrence(k_occ),
        "robin": robinhood_index(k_occ),
    }
