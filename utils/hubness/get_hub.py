from .metrics import score as score_hubness               
import numpy as np
import pandas as pd
from .knn_kocc import k_occurrence_ONLY
from .hub_score import hub_score

def get_hub_score(features,k_nn,dist_metric="sqeuclidean"):
    hubness = k_occurrence_ONLY(features, k=k_nn, metric=dist_metric)
    scores_all = hub_score(hubness,k=k_nn)
    return scores_all

def get_hub_occOnly(features,k_nn,dist_metric="sqeuclidean"):
    hubness = k_occurrence_ONLY(features, k=k_nn, metric=dist_metric)
    return hubness








