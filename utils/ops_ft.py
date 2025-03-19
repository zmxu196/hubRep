
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop
from sklearn.cluster import KMeans
from utils.ops_al import target_distribution
from utils.ops_ev import get_evaluation_results
import torch.optim as optim
# from models.CL_loss import CL_loss
from utils.hubness.get_hub import get_hub_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.neighbors import NearestNeighbors
from time import time

def training(args,logger,model, feature_list, learning_rate, weight_decay, num_epochs, sp_weight,labels):
    logger.info("###################### Start Training The Whole hubREP Model ######################")
    model.cuda()
    for i in range(model.num_views):
        feature_list[i] = feature_list[i].cuda()

    optimizer  = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay) 
    loss_function = nn.MSELoss()

    kmeans = KMeans(n_clusters=len(np.unique(labels)), n_init=5)

    for epoch in range(num_epochs):

        model.train()

        combined_feature, adj_wave_list, adj_bar_list,hidden_bar_list,_,pre_hidden_list, [norm_list,weight_list,_,loss_crossview_align_dict,_,loss_Hub_LOCAL_list,loss_Hub_UNIFORM_list] = model(feature_list,args,logger)

        optimizer.zero_grad()

        lossr_list = []
        losss_list = []
        for v in range(model.num_views):
            lossr_list.append(loss_function(pre_hidden_list[v], hidden_bar_list[v]))
            # losss_list.append(sp_weight * norm_list[v]* loss_function(adj_bar_list[v], adj_wave_list[v].to_dense().cuda()))
            losss_list.append(sp_weight * norm_list[v]* F.binary_cross_entropy(adj_bar_list[v].view(-1), adj_wave_list[v].to_dense().view(-1).cuda(),weight = (weight_list[v].cuda())))
        loss_lr = sum(lossr_list)/model.num_views
        loss_ls = sum(losss_list)/model.num_views  
        loss_grec = loss_ls + loss_lr

        loss_Hub_LOCAL =  sum(loss_Hub_LOCAL_list)/model.num_views
        loss_Hub_UNIFORM = sum(loss_Hub_UNIFORM_list)/model.num_views
        loss_pairview = torch.mean(torch.stack(list(loss_crossview_align_dict.values())))

        loss = loss_grec+ args.ft_Hub_weight * ((1-args.LAMBDA)* loss_Hub_UNIFORM + args.LAMBDA * loss_Hub_LOCAL) + args.ft_CL_weight * loss_pairview

        loss.backward()
        optimizer.step(closure=None)
 
        loss_value = float(loss.item())
 
        if (epoch+1) % 50 == 0:
            with torch.no_grad():
                logger.info("Epoch {:4d} | Total Loss: {:.5f}".format(epoch + 1,loss_value))

    model.eval()
    combined_feature, _, _,_,_,_, _ = model(feature_list,args,logger)

    y_pred = kmeans.fit_predict(combined_feature.detach().cpu())

    hub_VIEW_list_list = []
    for v in range(model.num_views):
        hub_VIEW_list = get_hub_score(pre_hidden_list[v],args.hub_k,dist_metric="sqeuclidean") 
        hub_VIEW_list_list.append(hub_VIEW_list)

    return y_pred, hub_VIEW_list_list
