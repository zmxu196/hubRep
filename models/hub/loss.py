import torch as th
from torch import nn


class HubAlignLoss(nn.Module):
    def __init__(self, kappa):
        super(HubAlignLoss, self).__init__()
        self.kappa = kappa

    def forward(self,embeddings,p_value):

        loss = -1 * (self.kappa * p_value[0] * (embeddings @ embeddings.transpose(0, 1))).sum(dim=(0, 1))
        return loss.mean()

class HubUniformLoss(nn.Module):
    def __init__(self, kappa):
        super(HubUniformLoss, self).__init__()
        self.kappa = kappa

    def forward(self, embeddings):
        logits = self.kappa * (embeddings @ embeddings.transpose(0, 1))
        loss = th.logsumexp(logits, dim=(0,1)) 
        return loss
