import math
import faiss
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn

import helpers


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

