import torch
import torch.autograd as autograd
from torch_geometric.data import Data, Batch


def dsm(energy_net, samples, sigma=1):
    samples.requires_grad_(True)
    vector = torch.randn_like(samples) * sigma
    perturbed_inputs = samples + vector
    logp = -energy_net(perturbed_inputs)
    dlogp = sigma ** 2 * autograd.grad(logp.sum(), perturbed_inputs, create_graph=True)[0]
    kernel = vector
    loss = torch.norm(dlogp + kernel, dim=-1) ** 2
    loss = loss.mean() / 2.

    return loss


def dsm_score_estimation(scorenet, samples, sigma=0.01):
    perturbed_samples = samples + torch.randn_like(samples) * sigma
    target = - 1 / (sigma ** 2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1).mean(dim=0)

    return loss


def anneal_dsm_score_estimation(scorenet, samples, labels, sigmas, anneal_power=2.):
    used_sigmas = sigmas[labels].view(samples.shape[0], 1)
    perturbed_samples = samples + torch.randn_like(samples) * used_sigmas.expand(samples.shape)
    target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples, used_sigmas)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    return loss.mean(dim=0)


def anneal_dsm_score_estimation_molecule(scorenet, data_batch, labels, sigmas, num_atoms, anneal_power=2.):
    used_sigmas = sigmas[labels].view(len(data_batch), 1)
    perturbed_positions = (data_batch.pos + torch.randn_like(data_batch.pos)).view(len(data_batch), num_atoms, 3) * used_sigmas.unsqueeze(-1).expand(len(data_batch), num_atoms, 3)

    data_batch_dict = dict(list(data_batch))
    pert_batch = Batch(x=data_batch_dict['x'],
      edge_index=data_batch_dict['edge_index'],
      y=data_batch_dict['y'],
      pos=perturbed_positions.view(-1,3),
      F=data_batch_dict['F'],
      batch=data_batch_dict['batch'],
      ptr=data_batch_dict['ptr'])
    
    target = - 1 / (used_sigmas.unsqueeze(-1).expand(len(data_batch), num_atoms, 3) ** 2) * (perturbed_positions - data_batch.pos.view(len(data_batch), num_atoms, 3))
    
    scores = scorenet(pert_batch, used_sigmas).view(len(data_batch), num_atoms, 3)

    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    return loss.mean(dim=0)
