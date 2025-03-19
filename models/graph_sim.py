import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class SimilarityAlignmentLoss(nn.Module): 
    def __init__(self, num_views, batch_size=256):
        super(SimilarityAlignmentLoss, self).__init__()
        self.n_views = num_views
        self.loss_function = nn.MSELoss()
        self.batch_size = batch_size

    @torch.no_grad()
    def _pairing_loss_two_views(self, Z1, Z2, weight1, weight2):
        epsilon = 1e-6
        num_samples = Z1.size(0)

        if num_samples < 5000:
            assert Z1.is_cuda and Z2.is_cuda
            weight_matrix = torch.outer(weight1 + epsilon, weight2 + epsilon)
            S1 = weight_matrix * torch.mm(Z1, Z1.t())
            S2 = weight_matrix * torch.mm(Z2, Z2.t())
            loss = self.loss_function(S1.view(-1), S2.view(-1)) / (num_samples * num_samples)
            return loss
        else:
            total_loss = torch.tensor(0.0, device=Z1.device) 
            batch_size = self.batch_size

            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                Z1_batch = Z1[i:end_idx]
                Z2_batch = Z2[i:end_idx]
                weight1_batch = weight1[i:end_idx]
                weight2_batch = weight2[i:end_idx]
                weight_matrix_batch = torch.outer(weight1_batch + epsilon, weight2_batch + epsilon)
                S1_batch = (weight_matrix_batch * torch.mm(Z1_batch, Z1_batch.t())).detach()
                S2_batch = (weight_matrix_batch * torch.mm(Z2_batch, Z2_batch.t())).detach()
                batch_loss = self.loss_function(S1_batch.view(-1), S2_batch.view(-1)) / (batch_size * batch_size)
                total_loss += batch_loss * (end_idx - i)

            return total_loss / num_samples

    def forward(self, view_features, hub_vlist):
        z = torch.stack(view_features, dim=0).cuda() 
        z = nn.functional.normalize(z, dim=-1, p=2)  
        losses = {}
        for u in range(self.n_views - 1):
            for v in range(u + 1, self.n_views):
                losses[f"{u}{v}"] = self._pairing_loss_two_views(z[u], z[v], hub_vlist[u], hub_vlist[v])

        return losses

