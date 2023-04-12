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

import ase
from ase.visualize import view
from ase.io import write

import os

__all__ = ['MD17Runner']


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


class MD17Runner():
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
    def anneal_langevin_dynamics(score, data_batch, num_atoms, sigmas, lr=0.1, n_steps_each=[], savepath=None, plot_freq=10):
        
        overall_counter = 0
        
        if savepath is not None:
            molecule_savepath = "%s/molecular_dynamics" % savepath
            if not os.path.exists(molecule_savepath):
                os.mkdir(molecule_savepath)

            test_ase_molecule = ase.Atoms(positions=data_batch.pos[data_batch.batch == 0], numbers=data_batch.x[data_batch.batch == 0].squeeze())
            write('{}/step_{:02d}.png'.format(molecule_savepath,overall_counter), test_ase_molecule)

        for s in range(sigmas.shape[0]):
            for i in range(n_steps_each[s]):
                current_lr = lr * (sigmas[s]) ** 2
                
                labels = s*torch.ones(len(data_batch),dtype=torch.long)
                used_sigmas = sigmas[labels].view(len(data_batch), 1)

                current_scores = score(data_batch, used_sigmas).view(len(data_batch), num_atoms, 3).detach()

                current_pos = data_batch.pos.view(len(data_batch), num_atoms, 3)

                new_pos = current_pos + current_lr / 2 * current_scores
                new_pos = new_pos + torch.randn_like(new_pos) * np.sqrt(current_lr)

                data_batch_dict = dict(list(data_batch))
                data_batch = Batch(x=data_batch_dict['x'],
                    edge_index=data_batch_dict['edge_index'],
                    y=data_batch_dict['y'],
                    pos=new_pos.view(-1,3),
                    F=data_batch_dict['F'],
                    batch=data_batch_dict['batch'],
                    ptr=data_batch_dict['ptr'])

                overall_counter += 1
                if savepath is not None and (overall_counter % plot_freq == 0):
                    test_ase_molecule = ase.Atoms(positions=data_batch.pos[data_batch.batch == 0], numbers=data_batch.x[data_batch.batch == 0].squeeze())
                    write('{}/step_{:02d}.png'.format(molecule_savepath,overall_counter), test_ase_molecule)

        return data_batch, overall_counter



    @staticmethod
    def langevin_dynamics_force(force_net, data_batch, num_atoms, lr=0.1, n_steps=100, savepath=None, plot_freq=10, overall_counter=0):
        
        overall_counter += 1
        
        if savepath is not None:
            molecule_savepath = "%s/molecular_dynamics" % savepath
            if not os.path.exists(molecule_savepath):
                os.mkdir(molecule_savepath)

            test_ase_molecule = ase.Atoms(positions=data_batch.pos[data_batch.batch == 0], numbers=data_batch.x[data_batch.batch == 0].squeeze())
            write('{}/step_{:02d}.png'.format(molecule_savepath,overall_counter), test_ase_molecule)

        for i in range(n_steps):

            current_forces = force_net(data_batch)[1].view(len(data_batch), num_atoms, 3).detach()

            current_pos = data_batch.pos.view(len(data_batch), num_atoms, 3).detach()

            new_pos = current_pos + lr / 2 * current_forces
            new_pos = new_pos + torch.randn_like(new_pos) * np.sqrt(lr)

            data_batch_dict = dict(list(data_batch))
            data_batch = Batch(x=data_batch_dict['x'],
                edge_index=data_batch_dict['edge_index'],
                y=data_batch_dict['y'],
                pos=new_pos.view(-1,3),
                F=data_batch_dict['F'],
                batch=data_batch_dict['batch'],
                ptr=data_batch_dict['ptr'])

            overall_counter += 1
            if savepath is not None and (overall_counter % plot_freq == 0):
                test_ase_molecule = ase.Atoms(positions=data_batch.pos[data_batch.batch == 0], numbers=data_batch.x[data_batch.batch == 0].squeeze())
                write('{}/step_{:02d}.png'.format(molecule_savepath,overall_counter), test_ase_molecule)

        return data_batch


    def train_molecule(self, num_molecules, molecule_name="benzene", lr_score=1e-5, lr_force=1e-3, hidden_units_score=128, hidden_units_force=16,
        sigma_start=1, sigma_end=1e-2, n_sigmas=2,
        train_force=False, score_and_force=True, savepath=None):
        
        np.random.seed(0)
        torch.manual_seed(121)

        if molecule_name == "benzene":
            md_dict = np.load("data/md17_benzene2017.npz")
            num_atoms = 12
        elif molecule_name == "ethanol":
            md_dict = np.load("data/md17_ethanol.npz")
            num_atoms = 9

        data_energies = md_dict['E']
        data_forces = md_dict['F']
        data_positions = md_dict['R']
        data_species = md_dict['z']

        molecule_graphs = []
        for i in range(num_molecules):
            
            dist_mat = get_distance_matrix(data_positions[i])
            
            edge_i, edge_j = (dist_mat * (dist_mat < 5.).astype(int)).nonzero()
            edges = np.concatenate([edge_i.reshape(-1,1), edge_j.reshape(-1,1)], axis=1)
            
            molecule_graphs.append(Data(x=torch.Tensor(data_species).unsqueeze(-1),
                                       edge_index=torch.tensor(edges, dtype=torch.long).t(),
                                       y=torch.tensor([data_energies[i]]),
                                       pos=torch.Tensor(data_positions[i]),
                                       F=torch.tensor(data_forces[i])))

        molecule_graphs_train = DataLoader(molecule_graphs[:int(len(molecule_graphs)*0.6)], batch_size=100, shuffle=True)
        molecule_graphs_val = DataLoader(molecule_graphs[int(len(molecule_graphs)*0.6):int(len(molecule_graphs)*0.8)], batch_size=100, shuffle=False)
        molecule_graphs_test = DataLoader(molecule_graphs[int(len(molecule_graphs)*0.8):], batch_size=100, shuffle=False)


        force_net = EnergyForceGraphAttention(node_embed_dim=hidden_units_force, smear_stop=5., smear_num_gaussians=50)

        sigmas = torch.Tensor(np.exp(np.linspace(np.log(sigma_start), np.log(sigma_end), n_sigmas)))
        print(sigmas)

        score_net = GraphConvolutionScoreNet(num_atoms=num_atoms, node_embed_dim=hidden_units_score, smear_stop=5., smear_num_gaussians=50)


        rho = self.rho
        force_optimizer = optim.Adam(force_net.parameters(), lr=lr_force)
        score_optimizer = optim.Adam(score_net.parameters(), lr=lr_score)
        if train_force:

            if savepath is not None and os.path.exists("%s/state_dict.pt" % savepath):
                force_net.load_state_dict(torch.load("%s/state_dict.pt" % savepath))
                force_net.eval()

            else:
                batch_training_losses_force = []
                training_losses_force = []
                validation_losses_force = []

                force_net.eval()
                val_running_loss = 0
                for valid_batch in molecule_graphs_val:
                    
                    energies_pred, forces_pred = force_net(valid_batch)
                    
                    loss_E = torch.nn.L1Loss(reduction='mean')(energies_pred.unsqueeze(-1), valid_batch.y)
                    loss_F = torch.nn.L1Loss(reduction='sum')(forces_pred, valid_batch.F) / valid_batch.num_graphs

                    loss = (rho*loss_E) + loss_F
                    
                    val_running_loss += loss.detach().item()*valid_batch.num_graphs
                validation_losses_force.append(val_running_loss)

                print("Pre-Train | Validation Loss {:.2e}".format(validation_losses_force[-1]))

                for epoch in range(self.n_epochs_force):

                    force_net.train()
                    running_loss = 0
                    for train_batch in molecule_graphs_train:
                        force_optimizer.zero_grad()

                        energies_pred, forces_pred = force_net(train_batch)

                        loss_E = torch.nn.L1Loss(reduction='mean')(energies_pred.unsqueeze(-1), train_batch.y)
                        loss_F = torch.nn.L1Loss(reduction='sum')(forces_pred, train_batch.F) / train_batch.num_graphs

                        loss = (rho*loss_E) + loss_F

                        loss.backward()
                        force_optimizer.step()
                        
                        running_loss += loss.detach().item()*train_batch.num_graphs
                        batch_training_losses_force.append(loss.detach().item()*train_batch.num_graphs)
                    training_losses_force.append(running_loss)

                    force_net.eval()
                    val_running_loss = 0
                    for valid_batch in molecule_graphs_val:
                        
                        energies_pred, forces_pred = force_net(valid_batch)
                        
                        loss_E = torch.nn.L1Loss(reduction='mean')(energies_pred.unsqueeze(-1), valid_batch.y)
                        loss_F = torch.nn.L1Loss(reduction='mean')(forces_pred, valid_batch.F) / valid_batch.num_graphs

                        loss = (rho*loss_E) + loss_F
                        
                        val_running_loss += loss.detach().item()*valid_batch.num_graphs
                    validation_losses_force.append(val_running_loss)

                    if (epoch+1) % 5 == 0:
                        print("Epoch {} | Train Loss {:.2e} | Validation Loss {:.2e}".format(epoch+1, training_losses_force[-1], validation_losses_force[-1]))

                if savepath is not None:
                    torch.save(force_net.state_dict(), "%s/state_dict.pt" % savepath)

                fig = plt.figure(figsize=(5,2))
                plt.plot(np.arange(1,self.n_epochs_force+1),training_losses_force, label="Training Loss")
                plt.plot(np.arange(0,self.n_epochs_force+1),validation_losses_force, label="Validation Loss")
                plt.xlabel('Training Epoch')
                plt.ylabel('Loss (Force Matching)')
                plt.legend()
                plt.tight_layout()
                if savepath is not None:
                    plt.savefig(savepath + "/force_training_loss.png")
                    plt.clf()
                    plt.close()
                else:
                    plt.show()


                fig = plt.figure(figsize=(5,2))
                plt.plot(batch_training_losses_force)
                plt.xlabel('Training Batch')
                plt.ylabel('Loss (Force Matching)')
                plt.tight_layout()
                if savepath is not None:
                    plt.savefig(savepath + "/force_batch_training_loss.png")
                    plt.clf()
                    plt.close()
                else:
                    plt.show()


            test_batch = next(enumerate(molecule_graphs_test))[1]
            test_batch_pos = test_batch.pos
            # noisy_test_batch_pos = torch.randn_like(test_batch_pos)
            noisy_test_batch_pos = test_batch_pos + torch.randn_like(test_batch_pos) * 1e-1

            test_batch_dict = dict(list(test_batch))
            test_batch = Batch(x=test_batch_dict['x'],
                edge_index=test_batch_dict['edge_index'],
                y=test_batch_dict['y'],
                pos=noisy_test_batch_pos.view(-1,3),
                F=test_batch_dict['F'],
                batch=test_batch_dict['batch'],
                ptr=test_batch_dict['ptr'])


            test_batch_out = self.langevin_dynamics_force(force_net, test_batch, num_atoms, lr=1e-4, n_steps=1000, savepath=savepath, plot_freq=10)


        elif score_and_force:

            if savepath is not None and os.path.exists("%s/state_dict_force.pt" % savepath):
                force_net.load_state_dict(torch.load("%s/state_dict_force.pt" % savepath))
                force_net.eval()

            else:
                batch_training_losses_force = []
                training_losses_force = []
                validation_losses_force = []

                force_net.eval()
                val_running_loss = 0
                for valid_batch in molecule_graphs_val:
                    
                    energies_pred, forces_pred = force_net(valid_batch)
                    
                    loss_E = torch.nn.L1Loss(reduction='mean')(energies_pred.unsqueeze(-1), valid_batch.y)
                    loss_F = torch.nn.L1Loss(reduction='sum')(forces_pred, valid_batch.F) / valid_batch.num_graphs

                    loss = (rho*loss_E) + loss_F
                    
                    val_running_loss += loss.detach().item()*valid_batch.num_graphs
                validation_losses_force.append(val_running_loss)

                print("Pre-Train | Validation Loss {:.2e}".format(validation_losses_force[-1]))

                for epoch in range(self.n_epochs_force):

                    force_net.train()
                    running_loss = 0
                    for train_batch in molecule_graphs_train:
                        force_optimizer.zero_grad()

                        energies_pred, forces_pred = force_net(train_batch)

                        loss_E = torch.nn.L1Loss(reduction='mean')(energies_pred.unsqueeze(-1), train_batch.y)
                        loss_F = torch.nn.L1Loss(reduction='sum')(forces_pred, train_batch.F) / train_batch.num_graphs

                        loss = (rho*loss_E) + loss_F

                        loss.backward()
                        force_optimizer.step()
                        
                        running_loss += loss.detach().item()*train_batch.num_graphs
                        batch_training_losses_force.append(loss.detach().item()*train_batch.num_graphs)
                    training_losses_force.append(running_loss)

                    force_net.eval()
                    val_running_loss = 0
                    for valid_batch in molecule_graphs_val:
                        
                        energies_pred, forces_pred = force_net(valid_batch)
                        
                        loss_E = torch.nn.L1Loss(reduction='mean')(energies_pred.unsqueeze(-1), valid_batch.y)
                        loss_F = torch.nn.L1Loss(reduction='mean')(forces_pred, valid_batch.F) / valid_batch.num_graphs

                        loss = (rho*loss_E) + loss_F
                        
                        val_running_loss += loss.detach().item()*valid_batch.num_graphs
                    validation_losses_force.append(val_running_loss)

                    if (epoch+1) % 5 == 0:
                        print("Epoch {} | Train Loss {:.2e} | Validation Loss {:.2e}".format(epoch+1, training_losses_force[-1], validation_losses_force[-1]))

                if savepath is not None:
                    torch.save(force_net.state_dict(), "%s/state_dict_force.pt" % savepath)

                fig = plt.figure(figsize=(5,2))
                plt.plot(np.arange(1,self.n_epochs_force+1),training_losses_force, label="Training Loss")
                plt.plot(np.arange(0,self.n_epochs_force+1),validation_losses_force, label="Validation Loss")
                plt.xlabel('Training Epoch')
                plt.ylabel('Loss (Force Matching)')
                plt.legend()
                plt.tight_layout()
                if savepath is not None:
                    plt.savefig(savepath + "/force_training_loss.png")
                    plt.clf()
                    plt.close()
                else:
                    plt.show()


                fig = plt.figure(figsize=(5,2))
                plt.plot(batch_training_losses_force)
                plt.xlabel('Training Batch')
                plt.ylabel('Loss (Force Matching)')
                plt.tight_layout()
                if savepath is not None:
                    plt.savefig(savepath + "/force_batch_training_loss.png")
                    plt.clf()
                    plt.close()
                else:
                    plt.show()


            if savepath is not None and os.path.exists("%s/state_dict_score.pt" % savepath):
                score_net.load_state_dict(torch.load("%s/state_dict_score.pt" % savepath))
                score_net.eval()

            else:
                batch_training_losses_score = []
                training_losses_score = []
                validation_losses_score = []

                score_net.eval()
                val_running_loss = 0
                for valid_batch in molecule_graphs_val:

                    if len(sigmas) > 1:
                        labels = torch.randint(low=0, high = n_sigmas-1, size = [len(valid_batch)])
                    else:
                        labels = torch.zeros(len(valid_batch),dtype=torch.long)
                    loss = anneal_dsm_score_estimation_molecule(score_net, valid_batch, labels, sigmas, num_atoms=num_atoms)

                    val_running_loss += loss.detach().item()*valid_batch.num_graphs
                validation_losses_score.append(val_running_loss)

                print("Pre-Train | Validation Loss {:.2e}".format(validation_losses_score[-1]))

                for epoch in range(self.n_epochs_score):

                    score_net.train()
                    running_loss = 0
                    for train_batch in molecule_graphs_train:
                        score_optimizer.zero_grad()

                        if len(sigmas) > 1:
                            labels = torch.randint(low=0, high = n_sigmas-1, size = [len(train_batch)])
                        else:
                            labels = torch.zeros(len(train_batch),dtype=torch.long)
                        loss = anneal_dsm_score_estimation_molecule(score_net, train_batch, labels, sigmas, num_atoms=num_atoms)

                        loss.backward()
                        score_optimizer.step()

                        running_loss += loss.detach().item()*train_batch.num_graphs
                        batch_training_losses_score.append(loss.detach().item()*train_batch.num_graphs)
                    training_losses_score.append(running_loss)

                    score_net.eval()
                    val_running_loss = 0
                    for valid_batch in molecule_graphs_val:

                        if len(sigmas) > 1:
                            labels = torch.randint(low=0, high = n_sigmas-1, size = [len(valid_batch)])
                        else:
                            labels = torch.zeros(len(valid_batch),dtype=torch.long)
                        loss = anneal_dsm_score_estimation_molecule(score_net, valid_batch, labels, sigmas, num_atoms=num_atoms)

                        val_running_loss += loss.detach().item()*valid_batch.num_graphs
                    validation_losses_score.append(val_running_loss)

                    if (epoch+1) % 1 == 0:
                        print("Epoch {} | Train Loss {:.2e} | Validation Loss {:.2e}".format(epoch+1, training_losses_score[-1], validation_losses_score[-1]))

                if savepath is not None:
                    torch.save(score_net.state_dict(), "%s/state_dict_score.pt" % savepath)

                fig = plt.figure(figsize=(5,2))
                plt.plot(np.arange(1,self.n_epochs_score+1),training_losses_score, label="Training Loss")
                plt.plot(np.arange(0,self.n_epochs_score+1),validation_losses_score, label="Validation Loss")
                plt.xlabel('Training Epoch')
                plt.ylabel('Loss (Score Matching)')
                plt.legend()
                plt.tight_layout()
                if savepath is not None:
                    plt.savefig(savepath + "/score_training_loss.png")
                    plt.clf()
                    plt.close()
                else:
                    plt.show()


                fig = plt.figure(figsize=(5,2))
                plt.plot(batch_training_losses_score)
                plt.xlabel('Training Batch')
                plt.ylabel('Loss (Score Matching)')
                plt.tight_layout()
                if savepath is not None:
                    plt.savefig(savepath + "/score_batch_training_loss.png")
                    plt.clf()
                    plt.close()
                else:
                    plt.show()



            test_batch = next(enumerate(molecule_graphs_test))[1]
            test_batch_pos = test_batch.pos
            noisy_test_batch_pos = torch.randn_like(test_batch_pos)
            # noisy_test_batch_pos = test_batch_pos + torch.randn_like(test_batch_pos) * 5e-1

            if savepath is not None:
                test_ase_molecule = ase.Atoms(positions=test_batch.pos[test_batch.batch == 0], numbers=test_batch.x[test_batch.batch == 0].squeeze())
                write('{}/real_molecule.png'.format(savepath), test_ase_molecule)

            test_batch_dict = dict(list(test_batch))
            test_batch = Batch(x=test_batch_dict['x'],
                edge_index=test_batch_dict['edge_index'],
                y=test_batch_dict['y'],
                pos=noisy_test_batch_pos.view(-1,3),
                F=test_batch_dict['F'],
                batch=test_batch_dict['batch'],
                ptr=test_batch_dict['ptr'])


            test_batch_out, overall_counter = self.anneal_langevin_dynamics(score_net, test_batch, num_atoms, sigmas, lr=1e-1, n_steps_each=[500,500], savepath=savepath, plot_freq=10)
            test_batch_out = self.langevin_dynamics_force(force_net, test_batch_out, num_atoms, lr=1e-4, n_steps=1000, savepath=savepath, plot_freq=10,
                overall_counter=overall_counter)

        else:

            if savepath is not None and os.path.exists("%s/state_dict.pt" % savepath):
                score_net.load_state_dict(torch.load("%s/state_dict.pt" % savepath))
                score_net.eval()

            else:
                batch_training_losses_score = []
                training_losses_score = []
                validation_losses_score = []

                score_net.eval()
                val_running_loss = 0
                for valid_batch in molecule_graphs_val:

                    if len(sigmas) > 1:
                        labels = torch.randint(low=0, high = n_sigmas-1, size = [len(valid_batch)])
                    else:
                        labels = torch.zeros(len(valid_batch),dtype=torch.long)
                    loss = anneal_dsm_score_estimation_molecule(score_net, valid_batch, labels, sigmas, num_atoms=num_atoms)

                    val_running_loss += loss.detach().item()*valid_batch.num_graphs
                validation_losses_score.append(val_running_loss)

                print("Pre-Train | Validation Loss {:.2e}".format(validation_losses_score[-1]))

                for epoch in range(self.n_epochs_score):

                    score_net.train()
                    running_loss = 0
                    for train_batch in molecule_graphs_train:
                        score_optimizer.zero_grad()

                        if len(sigmas) > 1:
                            labels = torch.randint(low=0, high = n_sigmas-1, size = [len(train_batch)])
                        else:
                            labels = torch.zeros(len(train_batch),dtype=torch.long)
                        loss = anneal_dsm_score_estimation_molecule(score_net, train_batch, labels, sigmas, num_atoms=num_atoms)

                        loss.backward()
                        score_optimizer.step()

                        running_loss += loss.detach().item()*train_batch.num_graphs
                        batch_training_losses_score.append(loss.detach().item()*train_batch.num_graphs)
                    training_losses_score.append(running_loss)

                    score_net.eval()
                    val_running_loss = 0
                    for valid_batch in molecule_graphs_val:

                        if len(sigmas) > 1:
                            labels = torch.randint(low=0, high = n_sigmas-1, size = [len(valid_batch)])
                        else:
                            labels = torch.zeros(len(valid_batch),dtype=torch.long)
                        loss = anneal_dsm_score_estimation_molecule(score_net, valid_batch, labels, sigmas, num_atoms=num_atoms)

                        val_running_loss += loss.detach().item()*valid_batch.num_graphs
                    validation_losses_score.append(val_running_loss)

                    if (epoch+1) % 1 == 0:
                        print("Epoch {} | Train Loss {:.2e} | Validation Loss {:.2e}".format(epoch+1, training_losses_score[-1], validation_losses_score[-1]))

                if savepath is not None:
                    torch.save(score_net.state_dict(), "%s/state_dict.pt" % savepath)

                fig = plt.figure(figsize=(5,2))
                plt.plot(np.arange(1,self.n_epochs_score+1),training_losses_score, label="Training Loss")
                plt.plot(np.arange(0,self.n_epochs_score+1),validation_losses_score, label="Validation Loss")
                plt.xlabel('Training Epoch')
                plt.ylabel('Loss (Score Matching)')
                plt.legend()
                plt.tight_layout()
                if savepath is not None:
                    plt.savefig(savepath + "/score_training_loss.png")
                    plt.clf()
                    plt.close()
                else:
                    plt.show()


                fig = plt.figure(figsize=(5,2))
                plt.plot(batch_training_losses_score)
                plt.xlabel('Training Batch')
                plt.ylabel('Loss (Score Matching)')
                plt.tight_layout()
                if savepath is not None:
                    plt.savefig(savepath + "/score_batch_training_loss.png")
                    plt.clf()
                    plt.close()
                else:
                    plt.show()


            test_batch = next(enumerate(molecule_graphs_test))[1]
            test_batch_pos = test_batch.pos
            noisy_test_batch_pos = torch.randn_like(test_batch_pos)
            # noisy_test_batch_pos = test_batch_pos + torch.randn_like(test_batch_pos) * 5e-1

            if savepath is not None:
                test_ase_molecule = ase.Atoms(positions=test_batch.pos[test_batch.batch == 0], numbers=test_batch.x[test_batch.batch == 0].squeeze())
                write('{}/real_molecule.png'.format(savepath), test_ase_molecule)

            test_batch_dict = dict(list(test_batch))
            test_batch = Batch(x=test_batch_dict['x'],
                edge_index=test_batch_dict['edge_index'],
                y=test_batch_dict['y'],
                pos=noisy_test_batch_pos.view(-1,3),
                F=test_batch_dict['F'],
                batch=test_batch_dict['batch'],
                ptr=test_batch_dict['ptr'])


            # test_batch_out = self.anneal_langevin_dynamics(score_net, test_batch, num_atoms, sigmas, lr=0.1, n_steps_each=[25,2000], savepath=savepath, plot_freq=10)
            # test_batch_out = self.anneal_langevin_dynamics(score_net, test_batch, num_atoms, sigmas, lr=0.1, n_steps_each=[25,500,1000], savepath=savepath, plot_freq=10)
            test_batch_out, _ = self.anneal_langevin_dynamics(score_net, test_batch, num_atoms, sigmas, lr=1e-1, n_steps_each=[500, 500], savepath=savepath, plot_freq=10)