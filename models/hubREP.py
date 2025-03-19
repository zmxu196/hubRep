
import torch as th
import torch.nn as nn
from layers.mlfpn_gcn import MLFPN_GCN
from utils.ops_al import dot_product_decode
from layers.mlfpn_fc import MLFPN_FC
from .encoder import Encoder
from utils.construct_graph import load_single_view_data
from .hub.HubLoss import HubLoss
from .graph_sim import SimilarityAlignmentLoss

class hubREP(nn.Module):

    def __init__(self,args, view_dims, encode_hidden_dim, pm_hidden_dims):
        super(hubREP, self).__init__()
        self.view_dims = view_dims
        self.pm_hidden_dims = pm_hidden_dims
        self.encode_hidden_dim = encode_hidden_dim
        self.num_views = len(view_dims)

        self.preliminary_module = nn.ModuleList()
        self.decoder_module = nn.ModuleList()
        self.pre_encoder  = nn.ModuleList()

        for i in range(self.num_views):
            temp_dims = []
            temp_dims.append(encode_hidden_dim)
            temp_dims.extend(pm_hidden_dims)
             # view-specifc encoder
            self.pre_encoder.append(Encoder(view_dims[i], encode_hidden_dim))
            # graph autoencoder
            self.preliminary_module.append(MLFPN_GCN(temp_dims, nn.ReLU()))
            self.decoder_module.append(MLFPN_FC(list(reversed(temp_dims)), nn.ReLU())) 

        self.Hub = HubLoss(args.KAPPA)
        self.paring_loss = SimilarityAlignmentLoss(self.num_views)
    def forward(self, feats, args,logger):

        hidden_list = []
        hidden_bar_list = []
        adj_bar_list = []

        pre_xrec_list = []
        pre_hidden_list = []
        pre_hidden_norm_list = []

        wave_list = []

        norm_list = []
        weight_list = []
        loss_Hub_LOCAL_list = []
        loss_Hub_UNIFORM_list = []
        hub_occ_view_list = []

        for i in range(self.num_views):

            pre_hidden = self.pre_encoder[i](feats[i])

            pre_hidden_norm = th.nn.functional.normalize(pre_hidden, p=2, dim=1)
            pre_hidden_list.append(pre_hidden)
            pre_hidden_norm_list.append(pre_hidden_norm)

            loss_local,loss_uniform = self.Hub(feats[i],pre_hidden_norm)

            loss_Hub_LOCAL_list.append(loss_local)
            loss_Hub_UNIFORM_list.append(loss_uniform)

            with th.no_grad():
                wave, hat,hub_occ,norm,weight_tensor = load_single_view_data(logger,pre_hidden_norm,k_nearest_neighbors=args.hub_k, prunning_one=True)

            wave_list.append(wave)
            norm_list.append(norm)  
            weight_list.append(weight_tensor)
            hub_occ_view_list.append(hub_occ)

            hidden = self.preliminary_module[i](pre_hidden_norm, hat.cuda())
            del hat
            hidden_list.append(hidden)

            hidden_bar = self.decoder_module[i](hidden)
            hidden_bar_list.append(hidden_bar)
            adj_bar = dot_product_decode(hidden)
            adj_bar_list.append(adj_bar)

        combined_feature = th.mean(th.stack(hidden_list), dim=0)
        crossview_alignloss = self.paring_loss(pre_hidden_norm_list,hub_occ_view_list)
        return combined_feature, wave_list, adj_bar_list,hidden_bar_list,pre_xrec_list,pre_hidden_list, [norm_list,weight_list,pre_hidden_norm_list,crossview_alignloss,hidden_list,loss_Hub_LOCAL_list,loss_Hub_UNIFORM_list]

