'''
Our project is based on @Project: DFP-GNN.
We sincerely appreciate the work of the authors.
@Project: DFP-GNN
@Time   : 2021/9/12 15:10
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail

'''

import os
import time
import torch
import random
import warnings
warnings.filterwarnings('ignore')
import argparse
import numpy as np
from models.hubREP import hubREP
from utils.ops_ft import training
from utils.ops_ev import get_evaluation_results
import wandb
import time
from utils.logger import get_logger
from utils.dataloader import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument('--cuda_device', type=str, default='0', help='The number of cuda device.')
    parser.add_argument('--n_repeated', type=int, default=5, help='Number of repeated experiments')
    parser.add_argument('--direction', type=str, default='./data/datasets/', help='direction of datasets')
    parser.add_argument('--dataset_name', type=str, default='BDGP', help='The dataset used for training/testing')
    parser.add_argument('--KAPPA', type=float, default=0.5, help='The hyperparameter for computing q')
    parser.add_argument('--LAMBDA', type=float, default=0.5 , help='The balance hyperparameter') 
    parser.add_argument('--pm_knns', type=int, default=10, help='k for kNN') 
    parser.add_argument('--hub_k', type=int, default=10, help='k for hubness calculation')
    parser.add_argument('--pre_hidden_dim', type=int, default=1024, help='latent space dimension')
    parser.add_argument('--pm_pruning_one', type=bool, default=True, help='prune')
    parser.add_argument('--pm_first_dim', type=int, default=512, help='dimension')
    parser.add_argument('--pm_second_dim', type=int, default=2048, help='dimension')
    parser.add_argument('--pm_third_dim', type=int, default=256, help='dimension')
    parser.add_argument('--ft_optimizer', type=str, default='Adam', help='optimizer type') 
    parser.add_argument('--ft_weight_decay', type=float, default=0., help='optimizer weight decay') 
    parser.add_argument('--ft_num_epochs', type=int, default=300, help='training epochs') 
    parser.add_argument('--ft_lr', type=float, default=0.0001 , help='The balance hyperparameter') 
    parser.add_argument('--ft_CL_weight', type=float, default=0.1, help='The balance hyperparameter') 
    parser.add_argument('--ft_Hub_weight', type=float, default=1, help='The balance hyperparameter' ) 
    parser.add_argument('--ft_sp_weight', type=float, default=0.00001 , help='The balance hyperparameter') 
    parser.add_argument('--seed', type=str, default=17204, help='The seed of randomness.')
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    logger = get_logger(args,timestamp)

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(args.seed)

    all_ACC = []
    all_NMI = []
    all_Purity = []
    all_ARI = []
    all_F = []
    all_P = []
    all_FT_TIME = []
    all_hub_VIEW_list = []

    for i in range(args.n_repeated):
        data = load_data(args.dataset_name)
        feature_list,num_view,feat_dims,label, n_clusters = get_data(data)

        args.num_classes = n_clusters
        args.num_views = num_view

        pm_hidden_dims = [args.pm_first_dim, args.pm_second_dim, args.pm_third_dim]

        model = hubREP(args,feat_dims,args.pre_hidden_dim,pm_hidden_dims)

        model.cuda()

        ft_begin_time = time.time()
        
        init_y_pred,hub_VIEW_list = training(args,logger,model, feature_list, learning_rate=args.ft_lr,weight_decay=args.ft_weight_decay,
                                               num_epochs=args.ft_num_epochs,sp_weight=args.ft_sp_weight, labels=label)
        ft_cost_time = time.time() - ft_begin_time

        ACC, NMI, Purity, ARI, P, R, F1 = get_evaluation_results(label, init_y_pred)

        logger.info('ACC:{:.5f}'.format(ACC))
        logger.info('NMI:{:.5f}'.format(NMI))
        logger.info('ARI:{:.5f}'.format(ARI))
        logger.info('Purity:{:.5f}:'.format(Purity))
        logger.info('P:{:.5f}'.format(P))
        logger.info('F1:{:.5f}'.format(F1))

        all_ACC.append(ACC)
        all_NMI.append(NMI)
        all_ARI.append(ARI)
        all_Purity.append(Purity)
        all_P.append(P)
        all_F.append(F1)
        all_FT_TIME.append(ft_cost_time)
        all_hub_VIEW_list.append(hub_VIEW_list)

    logger.info('------------------------------')
    logger.info
    logger.info("ACC: {:.4f} + std {:.4f}".format(np.mean(all_ACC), np.std(all_ACC)))
    logger.info("NMI: {:.4f} + std {:.4f}".format(np.mean(all_NMI), np.std(all_NMI)))
    logger.info("ARI: {:.4f} + std {:.4f}".format(np.mean(all_ARI) , np.std(all_ARI)))
    logger.info("Purity: {:.4f} + std {:.4f}".format(np.mean(all_Purity), np.std(all_Purity)))
    logger.info("P: {:.4f} + std {:.4f}".format(np.mean(all_P), np.std(all_P)))
    logger.info("F: {:.4f} + std {:.4f}".format(np.mean(all_F), np.std(all_F)))

    logger.handlers.clear()



if __name__ == '__main__':
    main()
