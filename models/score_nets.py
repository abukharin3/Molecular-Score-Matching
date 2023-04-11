import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_add_pool, GCNConv


class GraphConvolutionScoreNet(torch.nn.Module):
    def __init__(self, num_atoms, node_dim=3, node_embed_dim=16, smear_start=0.0, smear_stop=5.0, smear_num_gaussians=50):
        super().__init__()
        
        self.num_atoms = num_atoms
        self.node_dim = node_dim
        
        self.init_embed = torch.nn.Linear(node_dim, node_embed_dim)
        
        self.gconv1 = GCNConv(in_channels=node_embed_dim,
                              out_channels=node_embed_dim,
                              aggr='mean')
        
        self.gconv2 = GCNConv(in_channels=node_embed_dim,
                              out_channels=node_embed_dim,
                              aggr='mean')
        
        self.post_embed1 = torch.nn.Linear(node_embed_dim, node_embed_dim)
        self.post_embed2 = torch.nn.Linear(node_embed_dim, node_dim)
        
    def forward(self, data, sigmas):
        x, pos, edge_index = data.x, data.pos, data.edge_index

        edge_weight = torch.norm(pos[edge_index[0,:]] - pos[edge_index[1,:]],  dim=1, p=2).unsqueeze(-1)

        x_embed = F.softplus(self.init_embed(pos))
        x_gc1 = F.softplus(self.gconv1(x_embed, edge_index, edge_weight))
        x_gc2 = F.softplus(self.gconv2(x_gc1, edge_index, edge_weight))
        
        y = F.softplus(self.post_embed1(x_gc2))
        scores = self.post_embed2(y).view(data.num_graphs, self.num_atoms, self.node_dim)
        
        scores = scores / sigmas.unsqueeze(-1).expand((data.num_graphs, self.num_atoms, self.node_dim))
        
        return scores.view(data.num_graphs * self.num_atoms, self.node_dim)