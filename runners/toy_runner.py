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

from bgflow import DoubleWellEnergy
from bgflow import GaussianMCMCSampler
from bgflow.utils.types import assert_numpy
import matplotlib as mpl

import os

__all__ = ['ToyRunner']


class MoleculeSet(torch.utils.data.Dataset):
    def __init__(self, molecules):
        self.molecules = molecules
    
    def __len__(self):
        return len(self.molecules)
    
    def __getitem__(self, idx):
        return self.molecules[idx]


class ToyRunner():
    def __init__(self, n_epochs_score, n_epochs_force, rho=0.01, num_generated_samples=10000, hidden_units_score=128, hidden_units_force=128):

        self.n_epochs_score = n_epochs_score
        self.n_epochs_force = n_epochs_force
        
        self.hidden_units_score = hidden_units_score
        self.hidden_units_force = hidden_units_force
        
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



    def train_doublewell(self, num_samples=1000, conservative_force=True, train_force=False, score_and_force=True, savepath=None):
        # Set up double well
        dim=2
        target = DoubleWellEnergy(dim, c=0.5)
        self.energy = target

        np.random.seed(0)
        torch.manual_seed(0)
        init_state = torch.Tensor([[-2, 0], [2, 0]])
        init_state = torch.cat([init_state, torch.Tensor(init_state.shape[0], dim-2).normal_()], dim=-1)
        target_sampler = GaussianMCMCSampler(target, init_state=init_state)
        data = target_sampler.sample(num_samples)

        if conservative_force:
            force_net = EnergyForceNet(self.hidden_units_force)
        else:
            force_net = ForceNet(self.hidden_units_force)

        sigma_start = 1
        sigma_end   = 1e-2
        n_sigmas = 10
        sigmas = torch.Tensor(np.exp(np.linspace(np.log(sigma_start), np.log(sigma_end), n_sigmas)))

        score_net = ScoreNet(sigmas, self.hidden_units_score)

        idx = np.arange(num_samples)
        np.random.shuffle(idx)

        train_x = data[idx]
        trainset = MoleculeSet(train_x)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

        rho = self.rho
        force_optimizer = optim.Adam(force_net.parameters(), lr=1e-3)
        score_optimizer = optim.Adam(score_net.parameters(), lr=1e-3)
        if train_force:
            epoch_losses_force = []
            for epoch in range(self.n_epochs_force):
                running_loss = 0
                for i, batch in enumerate(trainloader):
                    force_optimizer.zero_grad()

                    force_labels = target.force(batch).detach()
                    energy_labels = target.force(batch).detach()

                    energies_pred, forces_pred = force_net(batch)
                    
                    if conservative_force:
                        loss_E = torch.nn.MSELoss()(energies_pred, energy_labels)
                        loss_F = torch.nn.MSELoss()(forces_pred, force_labels)

                        loss = (rho*loss_E) + loss_F

                    else:
                        loss = torch.nn.MSELoss()(forces_pred, force_labels)

                    loss.backward()
                    force_optimizer.step()
                    running_loss += loss.item()
                print("Epoch: {}, Total Loss: {}".format(epoch, running_loss))
                epoch_losses_force.append(running_loss)

            plt.plot(epoch_losses_force)
            plt.xlabel('Training Epoch')
            plt.ylabel('Training Loss (Force Matching)')
            if savepath is not None:
                plt.savefig(savepath + "/force_training_loss.png")
                plt.clf()
                plt.close()
            else:
                plt.show()


            self.visualize_doublewell(train_x, None, force_net, left_bound=-6.5, right_bound=6.5, sigmas=sigmas, force=True, savepath=savepath)

        elif score_and_force:
            epoch_losses_force = []
            for epoch in range(self.n_epochs_force):
                running_loss = 0
                for i, batch in enumerate(trainloader):
                    force_optimizer.zero_grad()

                    force_labels = target.force(batch).detach()
                    energy_labels = target.force(batch).detach()

                    energies_pred, forces_pred = force_net(batch)
                    
                    if conservative_force:
                        loss_E = torch.nn.MSELoss()(energies_pred, energy_labels)
                        loss_F = torch.nn.MSELoss()(forces_pred, force_labels)

                        loss = (rho*loss_E) + loss_F

                    else:
                        loss = torch.nn.MSELoss()(forces_pred, force_labels)

                    loss.backward()
                    force_optimizer.step()
                    running_loss += loss.item()
                print("Epoch: {}, Total Loss: {}".format(epoch, running_loss))
                epoch_losses_force.append(running_loss)

            plt.plot(epoch_losses_force)
            plt.xlabel('Training Epoch')
            plt.ylabel('Training Loss (Force Matching)')
            if savepath is not None:
                plt.savefig(savepath + "/force_training_loss.png")
                plt.clf()
                plt.close()
            else:
                plt.show()

            epoch_losses_score = []
            for epoch in range(self.n_epochs_score):
                running_loss = 0
                for i, batch in enumerate(trainloader):
                    score_optimizer.zero_grad()

                    labels = torch.randint(low=0, high = n_sigmas-1, size = [batch.shape[0]])
                    loss = anneal_dsm_score_estimation(score_net, batch, labels, sigmas)

                    loss.backward()
                    score_optimizer.step()
                    running_loss += loss.item()
                print("Epoch: {}, Total Loss: {}".format(epoch, running_loss))
                epoch_losses_score.append(running_loss)

            plt.plot(epoch_losses_score)
            plt.xlabel('Training Epoch')
            plt.ylabel('Training Loss (Score Matching)')
            if savepath is not None:
                plt.savefig(savepath + "/score_training_loss.png")
                plt.clf()
                plt.close()
            else:
                plt.show()

            self.visualize_doublewell(train_x, score_net, force_net, left_bound=-6.5, right_bound=6.5, sigmas=sigmas, force=False, score_and_force=True, savepath=savepath)

        else:
            epoch_losses_score = []
            for epoch in range(self.n_epochs_score):
                running_loss = 0
                for i, batch in enumerate(trainloader):
                    score_optimizer.zero_grad()

                    labels = torch.randint(low=0, high = n_sigmas-1, size = [batch.shape[0]])
                    loss = anneal_dsm_score_estimation(score_net, batch, labels, sigmas)

                    loss.backward()
                    score_optimizer.step()
                    running_loss += loss.item()
                print("Epoch: {}, Total Loss: {}".format(epoch, running_loss))
                epoch_losses_score.append(running_loss)

            plt.plot(epoch_losses_score)
            plt.xlabel('Training Epoch')
            plt.ylabel('Training Loss (Score Matching)')
            if savepath is not None:
                plt.savefig(savepath + "/score_training_loss.png")
                plt.clf()
                plt.close()
            else:
                plt.show()

            self.visualize_doublewell(train_x, score_net, None, left_bound=-6.5, right_bound=6.5, sigmas=sigmas, force=False, savepath=savepath)


    def doublewell_md_true(self, device=None, num_samples=1000):

        if not os.path.exists("results/toy_true_forces"):
            os.mkdir("results/toy_true_forces")

        # Set up double well
        dim=2
        target = DoubleWellEnergy(dim, c=0.5)
        self.energy = target

        left_bound=-2.75
        right_bound=2.75

        init_positions = torch.rand(10000, 2) * (right_bound - left_bound) + left_bound

        print("!", type(init_positions))
        self.plot_energy(self.energy, extent=(left_bound, right_bound))
        plt.scatter(init_positions[:, 0], init_positions[:, 1], s=0.1)
        plt.title("Sampled Data")
        plt.xlim([left_bound, right_bound])
        plt.ylim([left_bound, right_bound])
        plt.savefig("results/toy_true_forces/iter0.png")
        plt.clf()
        plt.close()

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

        forces = target.force(mesh.detach())
        mesh = mesh.detach().numpy()
        forces = forces.detach().numpy()

        plt.grid(False)
        plt.axis('off')
        plt.quiver(mesh[:, 0], mesh[:, 1], forces[:, 0], forces[:, 1], width=0.005)
        plt.scatter(init_positions[:, 0], init_positions[:, 1], s=0.1)
        plt.title('True Forces', fontsize=16)
        plt.axis('square')
        x = np.linspace(left_bound, right_bound, grid_size)
        y = np.linspace(left_bound, right_bound, grid_size)
        plt.show()

        masses = torch.ones(init_positions.shape[0]) * 1
        dynamics_driver = PhysicalLangevinDynamicsMonoatomic(init_positions=init_positions, masses=masses, dt=0.1, temperature=293)
        for i in range(1,500+1):
            dynamics_driver.true_step(target)

            final_positions = dynamics_driver.positions.detach().numpy()

            plt.scatter(final_positions[:, 0], final_positions[:, 1], s=0.1)
            # plt.axis('square')
            plt.title('Generated Data')
            plt.xlim([left_bound, right_bound])
            plt.ylim([left_bound, right_bound])
            plt.savefig("results/toy_true_forces/iter{:03d}.png".format(i))
            plt.clf()
            plt.close()




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