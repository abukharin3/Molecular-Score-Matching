import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_add_pool, GATv2Conv


class GaussianSmearing(torch.nn.Module):
    '''
    Borrowed From Schnet
    '''
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
    
    
class EnergyForceGraphAttention(torch.nn.Module):
    def __init__(self, node_dim=3, node_embed_dim=16, smear_start=0.0, smear_stop=5.0, smear_num_gaussians=50):
        super().__init__()
        self.init_embed = torch.nn.Linear(node_dim, node_embed_dim)
        
        self.gconv1 = GATv2Conv(in_channels=node_embed_dim,
                              out_channels=node_embed_dim,
                              edge_dim=smear_num_gaussians, heads=2, concat=False,
                              aggr='mean')
        
        self.gconv2 = GATv2Conv(in_channels=node_embed_dim,
                              out_channels=node_embed_dim,
                              edge_dim=smear_num_gaussians, heads=2, concat=False,
                              aggr='mean')
        
        self.post_embed1 = torch.nn.Linear(node_embed_dim, node_embed_dim)
        self.post_embed2 = torch.nn.Linear(node_embed_dim, 1)
        
        self.distance_expansion = GaussianSmearing(start=smear_start, stop=smear_stop, num_gaussians=smear_num_gaussians)
        
    def forward(self, data):
        x, pos, edge_index = data.x, data.pos, data.edge_index

        pos.requires_grad = True

        edge_weight = torch.norm(pos[edge_index[0,:]] - pos[edge_index[1,:]],  dim=1, p=2).unsqueeze(-1)
        edge_attr = self.distance_expansion(edge_weight)

        x_embed = F.softplus(self.init_embed(pos))
        x_gc1 = F.softplus(self.gconv1(x_embed, edge_index, edge_attr))
        x_gc2 = F.softplus(self.gconv2(x_gc1, edge_index, edge_attr))
        
        y = global_add_pool(x_gc2, data.batch)
        
        y = F.softplus(self.post_embed1(y))
        energy = self.post_embed2(y).view(-1)
        
        forces = -1*torch.autograd.grad(energy.sum(), pos, create_graph=True)[0]
        
        return energy, forces