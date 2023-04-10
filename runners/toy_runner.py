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
    def __init__(self, args, config):
        self.args = args
        self.config = config


    @staticmethod
    def langevin_dynamics(score, init, lr=0.1, step=100):
        for i in range(step):
            current_lr = lr
            init = init + current_lr / 2 * score(init).detach()
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


    def plot_energy(self, energy, extent=(-2.5, 2.5), resolution=100, dim=2):
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
        plt.show()


    def visualize_doublewell(self, data, model, left_bound=-1., right_bound=1., savefig=None, step=None, device=None, sigmas=None):
        print("!", type(data))
        self.plot_energy(self.energy, extent=(left_bound, right_bound))
        plt.scatter(data[:, 0], data[:, 1], s=0.1)
        plt.title("Sampled Data")
        plt.xlim([left_bound, right_bound])
        plt.ylim([left_bound, right_bound])
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

        scores = model(mesh.detach(), sigmas=sigmas[-1])
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
        plt.show()

        samples = torch.rand(10000, 2) * (right_bound - left_bound) + left_bound
        #samples = ToyRunner.langevin_dynamics(model, samples).detach().numpy()
        samples = ToyRunner.anneal_langevin_dynamics(model, samples, sigmas, lr=5e-2).detach().numpy()
        print("KL divergence:", kl_divergence(data, samples))
        plt.scatter(samples[:, 0], samples[:, 1], s=0.1)
        plt.axis('square')
        plt.title('Generated Data')
        plt.xlim([left_bound, right_bound])
        plt.ylim([left_bound, right_bound])
        plt.show()


    def train_doublewell(self, loss = "vr", num_samples=1000):
        # Set up double well
        dim=2
        target = DoubleWellEnergy(dim, c=0.5)
        self.energy = target

        init_state = torch.Tensor([[-2, 0], [2, 0]])
        init_state = torch.cat([init_state, torch.Tensor(init_state.shape[0], dim-2).normal_()], dim=-1)
        target_sampler = GaussianMCMCSampler(target, init_state=init_state)
        data = target_sampler.sample(num_samples)

        hidden_units = 128

        sigma_start = 1
        sigma_end   = 1e-2
        n_sigmas = 10
        sigmas = torch.Tensor(np.exp(np.linspace(np.log(sigma_start), np.log(sigma_end), n_sigmas)))

        score = ScoreNet(sigmas, hidden_units)

        batch_size = 128
        idx = np.arange(num_samples)
        np.random.shuffle(idx)

        train_x = data[idx]
        trainset = MoleculeSet(train_x)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(score.parameters(), lr=1e-4)
        

        for epoch in range(32):
            running_loss = 0
            for i, batch in enumerate(trainloader):
                optimizer.zero_grad()

                # loss, *_ = sliced_score_estimation_vr(score, batch, n_particles=1)
                # loss = dsm_score_estimation(score, batch, sigma=1e-1)
                labels = torch.randint(low=0, high = n_sigmas-1, size = [batch.shape[0]])
                loss = anneal_dsm_score_estimation(score, batch, labels, sigmas)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print("Epoch: {}, Total Loss: {}".format(epoch, running_loss))

        self.visualize_doublewell(train_x, score, left_bound=-3.5, right_bound=3.5, sigmas=sigmas)


    def train_force_doublewell(self, num_samples=1000):
        # Set up double well
        dim=2
        target = DoubleWellEnergy(dim, c=0.5)
        self.energy = target

        init_state = torch.Tensor([[-2, 0], [2, 0]])
        init_state = torch.cat([init_state, torch.Tensor(init_state.shape[0], dim-2).normal_()], dim=-1)
        target_sampler = GaussianMCMCSampler(target, init_state=init_state)
        
        data = target_sampler.sample(num_samples)
        data_energies = target.energy(data).detach() # [num_samples, 1]
        data_forces = target.force(data).detach() # [num_samples, 2]

        hidden_units = 128

        efn = EnergyForceNet(hidden_units)

        batch_size = 128
        idx = np.arange(num_samples)
        np.random.shuffle(idx)

        train_x = data[idx]
        train_energy = data_energies[idx]
        train_force = data_forces[idx]

        optimizer = optim.Adam(efn.parameters(), lr=1e-4)
        rho = 0.01
        
        num_batches = num_samples // batch_size
        for epoch in range(32):
            running_loss = 0
            for b in range(num_batches):
                optimizer.zero_grad()
                
                batch = train_x[b*batch_size:(b+1)*batch_size]
                
                energies_batch = train_energy[b*batch_size:(b+1)*batch_size]
                forces_batch = train_force[b*batch_size:(b+1)*batch_size]

                energies_pred, forces_pred = efn(batch)

                loss_E = torch.nn.MSELoss()(energies_pred, energies_batch)
                loss_F = torch.nn.MSELoss()(forces_pred, forces_batch)

                loss = (rho*loss_E) + loss_F
                
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print("Epoch: {}, Total Loss: {}".format(epoch, running_loss))

        # self.visualize_doublewell(train_x, score, left_bound=-3.5, right_bound=3.5, sigmas=sigmas)


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
        self.fc3 = nn.Linear(H, H)
        self.fc4 = nn.Linear(H, 1)

    def forward(self, x):

        x.requires_grad = True

        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc3(y))
        energy = self.fc4(y)

        force = -1*torch.autograd.grad(energy.sum(), x, create_graph=True)[0]

        return energy, force