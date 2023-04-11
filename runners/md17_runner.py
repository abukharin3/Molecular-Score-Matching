import logging
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from losses.sliced_sm import *
from losses.dsm import *
from dynamics.physical_langevin_dynamics import PhysicalLangevinDynamicsMonoatomic
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from losses.kl_divergence import kl_divergence
sns.set()
sns.set_style('white')

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_add_pool

from models.force_nets import EnergyForceGraphAttention
from models.score_nets import GraphConvolutionScoreNet

import os

__all__ = ['BenzeneRunner']


def get_distance_matrix(positions):
    '''
    Given the positions of atoms in a structure, compute shortest
    pairwise distances between each atom.
    
    Parameters
    ----------
    - positions : N x 3 array containining 3D position of N atoms
        
    Returns
    -------
    - distance_matrix : N x N array of shortest pairwise distances between
        atoms in a structure
    '''
    distance_matrix = np.zeros((positions.shape[0],positions.shape[0]))
    # iterate through all pairs of atoms only once per pair
    for i in range(positions.shape[0]):
        for j in range(i+1, positions.shape[0]):
            distance_matrix[i,j] = ((positions[i] - positions[j])**2).sum()**(0.5)
            distance_matrix[j,i] = distance_matrix[i,j]
                
    return distance_matrix


class MoleculeSet(torch.utils.data.Dataset):
    def __init__(self, molecules):
        self.molecules = molecules
    
    def __len__(self):
        return len(self.molecules)
    
    def __getitem__(self, idx):
        return self.molecules[idx]


class BenzeneRunner():
    def __init__(self, n_epochs_score, n_epochs_force, rho=0.01, num_generated_samples=10000):

        self.n_epochs_score = n_epochs_score
        self.n_epochs_force = n_epochs_force
        
        self.rho = rho
        self.num_generated_samples = num_generated_samples


    @staticmethod
    def simple_langevin_dynamics(score, init, lr=0.1, step=100):
        for i in range(step):
            current_lr = lr
            init = (init + current_lr / 2 * score(init)[1].detach()).detach()
            init = init + torch.randn_like(init) * np.sqrt(current_lr)
        return init


    @staticmethod
    def anneal_langevin_dynamics(score, init, sigmas, lr=0.1, n_steps_each=100):
        for sigma in sigmas:
            for i in range(n_steps_each):
                current_lr = lr * (sigma) ** 2
                init = init + current_lr / 2 * score(init, sigma).detach()
                init = init + torch.randn_like(init) * np.sqrt(current_lr)

        return init


    @staticmethod
    def physical_dynamics(force_net, init, step=100):
        masses = torch.ones(init.shape[0])
        dynamics_driver = PhysicalLangevinDynamicsMonoatomic(init_positions=init, masses=masses, dt=1, temperature=300)
        for i in range(step):
            dynamics_driver.step(force_net)
        return dynamics_driver.positions


    def plot_energy(self, energy, extent=(-2.5, 2.5), resolution=100, dim=2, savepath=None):
        """ Plot energy functions in 2D """
        xs = torch.meshgrid([torch.linspace(*extent, resolution) for _ in range(2)])
        xs = torch.stack(xs, dim=-1).view(-1, 2)
        xs = torch.cat([
            xs,
            torch.Tensor(xs.shape[0], dim - xs.shape[-1]).zero_()
        ], dim=-1)
        us = energy.energy(xs).view(resolution, resolution)
        us = torch.exp(-us)
        plt.imshow(assert_numpy(us).T, extent=extent * 2)
        plt.xlim([extent[0], extent[1]])
        plt.ylim([extent[0], extent[1]])
        plt.title("Data Density")
        if savepath is not None:
            plt.savefig(savepath + "/true_energy.png")
            plt.clf()
            plt.close()
        else:
            plt.show()


    def visualize_doublewell(self, data, smodel, fmodel, left_bound=-1., right_bound=1.,
        step=None, device=None, sigmas=None, force=True, score_and_force=False, savepath=None):

        self.plot_energy(self.energy, extent=(left_bound, right_bound), savepath=savepath)
        
        plt.scatter(data[:, 0], data[:, 1], s=0.1)
        plt.title("Training Data")
        plt.axis('square')
        plt.xlim([left_bound, right_bound])
        plt.ylim([left_bound, right_bound])
        if savepath is not None:
            plt.savefig(savepath + "/training_data.png")
            plt.clf()
            plt.close()
        else:
            plt.show()

        grid_size = 20
        mesh = []
        x = np.linspace(left_bound, right_bound, grid_size)
        y = np.linspace(left_bound, right_bound, grid_size)
        for i in x:
            for j in y:
                mesh.append(np.asarray([i, j]))

        mesh = np.stack(mesh, axis=0)
        mesh = torch.from_numpy(mesh).float()
        if device is not None:
            mesh = mesh.to(device)

        if not score_and_force:
            if force:
                _, scores = fmodel(mesh.detach())
            else:
                scores = smodel(mesh.detach(), sigmas=sigmas[-1])
            mesh = mesh.detach().numpy()
            scores = scores.detach().numpy()

            plt.grid(False)
            plt.axis('off')
            plt.quiver(mesh[:, 0], mesh[:, 1], scores[:, 0], scores[:, 1], width=0.005)
            plt.scatter(data[:, 0], data[:, 1], s=0.1)
            plt.title('Estimated scores', fontsize=16)
            plt.axis('square')
            x = np.linspace(left_bound, right_bound, grid_size)
            y = np.linspace(left_bound, right_bound, grid_size)
            if savepath is not None:
                plt.savefig(savepath + "/estimated_scores.png")
                plt.clf()
                plt.close()
            else:
                plt.show()

            torch.manual_seed(0)
            samples = torch.rand(self.num_generated_samples, 2) * (right_bound - left_bound) + left_bound
            
            if force:
                samples = ToyRunner.simple_langevin_dynamics(fmodel, samples, lr=5e-2).detach().numpy()
            else:
                samples = ToyRunner.anneal_langevin_dynamics(smodel, samples, sigmas, lr=5e-2).detach().numpy()
            
            dim = 2
            target = DoubleWellEnergy(dim, c=0.5)
            self.energy = target
            np.random.seed(0)
            torch.manual_seed(0)
            init_state = torch.Tensor([[-2, 0], [2, 0]])
            init_state = torch.cat([init_state, torch.Tensor(init_state.shape[0], dim - 2).normal_()], dim=-1)
            target_sampler = GaussianMCMCSampler(target, init_state=init_state)
            data = target_sampler.sample(self.num_generated_samples)

            kl_err = kl_divergence(data, samples)

            print("KL divergence:", kl_err)

            plt.scatter(samples[:, 0], samples[:, 1], s=0.1)
            plt.axis('square')
            plt.title('Generated Data | KL Divergence {:.2f}'.format(kl_err))
            plt.xlim([left_bound, right_bound])
            plt.ylim([left_bound, right_bound])
            if savepath is not None:
                plt.savefig(savepath + "/generated_data.png")
                plt.clf()
                plt.close()
            else:
                plt.show()



        else:
            _, force_scores = fmodel(mesh.detach())
            sm_scores = smodel(mesh.detach(), sigmas=sigmas[-1])

            mesh = mesh.detach().numpy()
            force_scores = force_scores.detach().numpy()
            sm_scores = sm_scores.detach().numpy()

            plt.grid(False)
            plt.axis('off')
            plt.quiver(mesh[:, 0], mesh[:, 1], sm_scores[:, 0], sm_scores[:, 1], width=0.005, color='red', label="Score Match Field")
            plt.quiver(mesh[:, 0], mesh[:, 1], force_scores[:, 0], force_scores[:, 1], width=0.005, color='blue', label="Force Match Field")
            plt.legend(bbox_to_anchor=(1.0, 0.05))
            plt.scatter(data[:, 0], data[:, 1], s=0.1)
            plt.title('Estimated scores', fontsize=16)
            plt.axis('square')
            x = np.linspace(left_bound, right_bound, grid_size)
            y = np.linspace(left_bound, right_bound, grid_size)
            if savepath is not None:
                plt.savefig(savepath + "/estimated_scores.png")
                plt.clf()
                plt.close()
            else:
                plt.show()

            torch.manual_seed(0)
            samples = torch.rand(self.num_generated_samples, 2) * (right_bound - left_bound) + left_bound

            samples = ToyRunner.anneal_langevin_dynamics(smodel, samples, sigmas[:], lr=5e-2, n_steps_each=75).detach()
            samples = ToyRunner.simple_langevin_dynamics(fmodel, samples, lr=5e-2, step=25).detach().numpy()
            
            dim = 2
            target = DoubleWellEnergy(dim, c=0.5)
            self.energy = target
            np.random.seed(0)
            torch.manual_seed(0)
            init_state = torch.Tensor([[-2, 0], [2, 0]])
            init_state = torch.cat([init_state, torch.Tensor(init_state.shape[0], dim - 2).normal_()], dim=-1)
            target_sampler = GaussianMCMCSampler(target, init_state=init_state)
            data = target_sampler.sample(self.num_generated_samples)

            kl_err = kl_divergence(data, samples)

            print("KL divergence:", kl_err)

            plt.scatter(samples[:, 0], samples[:, 1], s=0.1)
            plt.axis('square')
            plt.title('Generated Data | KL Divergence {:.2f}'.format(kl_err))
            plt.xlim([left_bound, right_bound])
            plt.ylim([left_bound, right_bound])
            if savepath is not None:
                plt.savefig(savepath + "/generated_data.png")
                plt.clf()
                plt.close()
            else:
                plt.show()



    def train_benzene(self, num_molecules, train_force=False, score_and_force=True, savepath=None):
        
        np.random.seed(0)
        torch.manual_seed(0)

        md_benzene = np.load("data/md17_benzene2017.npz")

        data_energies = md_benzene['E']
        data_forces = md_benzene['F']
        data_positions = md_benzene['R']
        data_species = md_benzene['z']

        benzene_graphs = []
        for i in range(num_molecules):
            
            dist_mat = get_distance_matrix(data_positions[i])
            
            edge_i, edge_j = (dist_mat * (dist_mat < 5.).astype(int)).nonzero()
            edges = np.concatenate([edge_i.reshape(-1,1), edge_j.reshape(-1,1)], axis=1)
            
            benzene_graphs.append(Data(x=torch.Tensor(data_species).unsqueeze(-1),
                                       edge_index=torch.tensor(edges, dtype=torch.long).t(),
                                       y=torch.tensor([data_energies[i]]),
                                       pos=torch.Tensor(data_positions[i]),
                                       F=torch.tensor(data_forces[i])))

        benzene_graphs_train = DataLoader(benzene_graphs[:int(len(benzene_graphs)*0.6)], batch_size=100, shuffle=True)
        benzene_graphs_val = DataLoader(benzene_graphs[int(len(benzene_graphs)*0.6):int(len(benzene_graphs)*0.8)], batch_size=100, shuffle=False)
        benzene_graphs_test = DataLoader(benzene_graphs[int(len(benzene_graphs)*0.8):], batch_size=100, shuffle=False)


        force_net = EnergyForceGraphAttention(smear_stop=5., smear_num_gaussians=50)


        sigma_start = 1
        sigma_end   = 1e-2
        n_sigmas = 10
        sigmas = torch.Tensor(np.exp(np.linspace(np.log(sigma_start), np.log(sigma_end), n_sigmas)))

        score_net = GraphConvolutionScoreNet(num_atoms=12)


        rho = self.rho
        force_optimizer = optim.Adam(force_net.parameters(), lr=1e-3)
        score_optimizer = optim.Adam(score_net.parameters(), lr=1e-3)
        if train_force:
            training_losses_force = []
            validation_losses_force = []
            for epoch in range(self.n_epochs_force):

                force_net.train()
                running_loss = 0
                for train_batch in benzene_graphs_train:
                    force_optimizer.zero_grad()

                    energies_pred, forces_pred = force_net(train_batch)

                    loss_E = torch.nn.L1Loss(reduction='mean')(energies_pred.unsqueeze(-1), train_batch.y)
                    loss_F = torch.nn.L1Loss(reduction='sum')(forces_pred, train_batch.F) / train_batch.num_graphs

                    loss = (rho*loss_E) + loss_F

                    loss.backward()
                    force_optimizer.step()
                    
                    running_loss += loss.detach().item()*train_batch.num_graphs
                training_losses_force.append(running_loss)

                force_net.eval()
                val_running_loss = 0
                for valid_batch in benzene_graphs_val:
                    
                    energies_pred, forces_pred = force_net(valid_batch)
                    
                    loss_E = torch.nn.L1Loss(reduction='mean')(energies_pred.unsqueeze(-1), valid_batch.y)
                    loss_F = torch.nn.L1Loss(reduction='sum')(forces_pred, valid_batch.F) / valid_batch.num_graphs

                    loss = (rho*loss_E) + loss_F
                    
                    val_running_loss += loss.item()*valid_batch.num_graphs
                validation_losses_force.append(val_running_loss)

                if (epoch+1) % 5 == 0:
                    print("Epoch {} | Train Loss {:.2e} | Validation Loss {:.2e}".format(epoch+1, training_losses_force[-1], validation_losses_force[-1]))

            fig = plt.figure(figsize=(5,2))
            plt.plot(training_losses_force, label="Training Loss")
            plt.plot(validation_losses_force, label="Validation Loss")
            plt.xlabel('Training Epoch')
            plt.ylabel('Loss (Force Matching)')
            plt.legend()
            if savepath is not None:
                plt.savefig(savepath + "/force_training_loss.png")
                plt.clf()
                plt.close()
            else:
                plt.show()


            # self.visualize_doublewell(train_x, None, force_net, left_bound=-6.5, right_bound=6.5, sigmas=sigmas, force=True, savepath=savepath)

        elif score_and_force:
            training_losses_force = []
            validation_losses_force = []
            for epoch in range(self.n_epochs_force):

                force_net.train()
                running_loss = 0
                for train_batch in benzene_graphs_train:
                    force_optimizer.zero_grad()

                    energies_pred, forces_pred = force_net(train_batch)

                    loss_E = torch.nn.L1Loss(reduction='mean')(energies_pred.unsqueeze(-1), train_batch.y)
                    loss_F = torch.nn.L1Loss(reduction='sum')(forces_pred, train_batch.F) / train_batch.num_graphs

                    loss = (rho*loss_E) + loss_F

                    loss.backward()
                    force_optimizer.step()
                    
                    running_loss += loss.detach().item()*train_batch.num_graphs
                training_losses_force.append(running_loss)

                force_net.eval()
                val_running_loss = 0
                for valid_batch in benzene_graphs_val:
                    
                    energies_pred, forces_pred = force_net(valid_batch)
                    
                    loss_E = torch.nn.L1Loss(reduction='mean')(energies_pred.unsqueeze(-1), valid_batch.y)
                    loss_F = torch.nn.L1Loss(reduction='sum')(forces_pred, valid_batch.F) / valid_batch.num_graphs

                    loss = (rho*loss_E) + loss_F
                    
                    val_running_loss += loss.item()*valid_batch.num_graphs
                validation_losses_force.append(val_running_loss)

                if (e+1) % report_rate == 0:
                    print("Epoch {} | Train Loss {:.2e} | Validation Loss {:.2e}".format(e+1, training_losses_force[-1], validation_losses_force[-1]))

            fig = plt.figure(figsize=(5,2))
            plt.plot(training_losses_force, label="Training Loss")
            plt.plot(validation_losses_force, label="Validation Loss")
            plt.xlabel('Training Epoch')
            plt.ylabel('Loss (Force Matching)')
            plt.legend()
            if savepath is not None:
                plt.savefig(savepath + "/force_training_loss.png")
                plt.clf()
                plt.close()
            else:
                plt.show()


            training_losses_score = []
            validation_losses_score = []
            for epoch in range(self.n_epochs_score):

                score_net.train()
                running_loss = 0
                for train_batch in benzene_graphs_train:
                    score_optimizer.zero_grad()

                    labels = torch.randint(low=0, high = n_sigmas-1, size = [len(train_batch)])
                    loss = anneal_dsm_score_estimation_molecule(score_net, train_batch, labels, sigmas, num_atoms=12)

                    loss.backward()
                    score_optimizer.step()

                    running_loss += loss.detach().item()*train_batch.num_graphs
                training_losses_score.append(running_loss)

                score_net.eval()
                val_running_loss = 0
                for valid_batch in benzene_graphs_val:

                    labels = torch.randint(low=0, high = n_sigmas-1, size = [len(valid_batch)])
                    loss = anneal_dsm_score_estimation_molecule(score_net, valid_batch, labels, sigmas, num_atoms=12)

                    loss.backward()

                    val_running_loss += loss.item()*valid_batch.num_graphs
                validation_losses_score.append(val_running_loss)

                if (epoch+1) % 5 == 0:
                    print("Epoch {} | Train Loss {:.2e} | Validation Loss {:.2e}".format(epoch+1, training_losses_score[-1], validation_losses_score[-1]))

            fig = plt.figure(figsize=(5,2))
            plt.plot(training_losses_score, label="Training Loss")
            plt.plot(validation_losses_score, label="Validation Loss")
            plt.xlabel('Training Epoch')
            plt.ylabel('Loss (Score Matching)')
            plt.legend()
            if savepath is not None:
                plt.savefig(savepath + "/score_training_loss.png")
                plt.clf()
                plt.close()
            else:
                plt.show()

            # self.visualize_doublewell(train_x, score_net, force_net, left_bound=-6.5, right_bound=6.5, sigmas=sigmas, force=False, score_and_force=True, savepath=savepath)

        else:
            training_losses_score = []
            validation_losses_score = []
            for epoch in range(self.n_epochs_score):

                score_net.train()
                running_loss = 0
                for train_batch in benzene_graphs_train:
                    score_optimizer.zero_grad()

                    labels = torch.randint(low=0, high = n_sigmas-1, size = [len(train_batch)])
                    loss = anneal_dsm_score_estimation_molecule(score_net, train_batch, labels, sigmas, num_atoms=12)

                    loss.backward()
                    score_optimizer.step()

                    running_loss += loss.detach().item()*train_batch.num_graphs
                training_losses_score.append(running_loss)

                score_net.eval()
                val_running_loss = 0
                for valid_batch in benzene_graphs_val:

                    labels = torch.randint(low=0, high = n_sigmas-1, size = [len(valid_batch)])
                    loss = anneal_dsm_score_estimation_molecule(score_net, valid_batch, labels, sigmas, num_atoms=12)

                    loss.backward()

                    val_running_loss += loss.item()*valid_batch.num_graphs
                validation_losses_score.append(val_running_loss)

                if (epoch+1) % 5 == 0:
                    print("Epoch {} | Train Loss {:.2e} | Validation Loss {:.2e}".format(epoch+1, training_losses_score[-1], validation_losses_score[-1]))

            fig = plt.figure(figsize=(5,2))
            plt.plot(training_losses_score, label="Training Loss")
            plt.plot(validation_losses_score, label="Validation Loss")
            plt.xlabel('Training Epoch')
            plt.ylabel('Loss (Score Matching)')
            plt.legend()
            if savepath is not None:
                plt.savefig(savepath + "/score_training_loss.png")
                plt.clf()
                plt.close()
            else:
                plt.show()

            # self.visualize_doublewell(train_x, score_net, None, left_bound=-6.5, right_bound=6.5, sigmas=sigmas, force=False, savepath=savepath)




class ScoreNet(nn.Module):
    def __init__(self, sigmas, H=128):
        super().__init__()
        self.sigmas = sigmas
        self.fc1 = nn.Linear(2, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, H)
        self.fc4 = nn.Linear(H, 2)

    def forward(self, x, sigmas):

        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc3(y))
        y = self.fc4(y) / sigmas.expand(x.shape)

        return y


class EnergyForceNet(nn.Module):
    def __init__(self, H=128):
        super().__init__()
        self.fc1 = nn.Linear(2, H)
        self.fc2 = nn.Linear(H, H)
        # self.fc3 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, 1)

    def forward(self, x):

        x.requires_grad = True

        y = nn.Softplus()(self.fc1(x))
        y = nn.Softplus()(self.fc2(y))
        # y = nn.Softplus()(self.fc3(y))
        energy = self.fc3(y)

        force = -1*torch.autograd.grad(energy.sum(), x, create_graph=True)[0]

        return energy, force


class ForceNet(nn.Module):
    def __init__(self, H=128):
        super().__init__()
        self.fc1 = nn.Linear(2, H)
        self.fc2 = nn.Linear(H, H)
        # self.fc3 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, 2)

    def forward(self, x):

        y = nn.Softplus()(self.fc1(x))
        y = nn.Softplus()(self.fc2(y))
        # y = nn.Softplus()(self.fc3(y))
        force = self.fc3(y)

        return None, force