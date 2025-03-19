import torch as th
from torch import nn
# from pca import get_pca_weights
from .util import x2p
from .loss import HubAlignLoss, HubUniformLoss


class HubLoss(nn.Module):
    def __init__(self,kappa, *, init="pca", out_dims=1024, initial_dims=None,  perplexity=45, re_norm=True,
                 eps=1e-12, p_sim="vmf", p_rel_tol=1e-2, p_abs_tol=None, p_betas=(None, None), pca_mode="base",
                 pca_weights=None, learning_rate=1e-1, n_iter=50, loss_weights=(0.2,)):
        super(HubLoss, self).__init__()

        self.re_norm = re_norm
        self.eps = th.tensor(eps).cuda()
        self.kappa = kappa
        # print('self.kappa',self.kappa)
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.perplexity = perplexity
        self.p_sim = p_sim
        self.p_rel_tol = p_rel_tol
        self.p_abs_tol = p_abs_tol
        self.eps = th.tensor(eps).cuda()
        self.p_betas = p_betas

        self.out_dims = 1024
        self.HubAlignLoss = HubAlignLoss(kappa=self.kappa)
        self.HubUniformLoss = HubUniformLoss(kappa=self.kappa)

    def forward(self,inputs,hidden):
        p_value = x2p(inputs, perplexity=self.perplexity, sim=self.p_sim, rel_tol=self.p_rel_tol,
                     abs_tol=self.p_abs_tol, eps=self.eps, betas=self.p_betas)
        loss_local = self.HubAlignLoss(hidden,p_value)
        loss_uniform = self.HubUniformLoss(hidden)
        # loss = loss_local + loss_uniform
        return loss_local,loss_uniform
        