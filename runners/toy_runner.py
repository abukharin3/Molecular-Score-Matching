import logging
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from losses.sliced_sm import *
from losses.dsm import *
from models.gmm import GMM, Gaussian, GMMDist, Square, GMMDistAnneal
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

__all__ = ['ToyRunner']


class MoleculeSet(torch.utils.data.Dataset):
    def __init__(self, molecules):
        self.molecules = molecules
    
    def __len__(self):
        return len(self.molecules)
    
    def __getitem__(self, idx):
        return self.molecules[idx]


class Ring():
    def __init__(self, radius, width):
        self.radius = radius
        self.width = width
        self.r_dist = Normal(loc=radius, scale=width)

    def sample(self, sample_shape):
        theta = torch.rand(sample_shape) * np.pi * 2
        r = self.r_dist.sample(sample_shape)
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        return torch.stack([x, y], dim=-1)

    def log_prob(self, inputs):
        r = torch.norm(inputs, dim=-1)
        return self.r_dist.log_prob(r) - torch.log(r * np.pi * 2)


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


    @staticmethod
    def visualize(teacher, model, left_bound=-1., right_bound=1., savefig=None, step=None, device=None):
        mesh = []
        grid_size = 100
        x = np.linspace(left_bound, right_bound, grid_size)
        y = np.linspace(left_bound, right_bound, grid_size)
        for i in x:
            for j in y:
                mesh.append(np.asarray([i, j]))

        mesh = np.stack(mesh, axis=0)
        mesh = torch.from_numpy(mesh).float()
        if device is not None:
            mesh = mesh.to(device)

        logp_true = teacher.log_prob(mesh)
        logp_true = logp_true.view(grid_size, grid_size).exp()

        plt.grid(False)
        plt.axis('off')
        plt.imshow(np.flipud(logp_true.cpu().numpy()), cmap='inferno')

        plt.title('Data density', fontsize=16)

        if savefig is not None:
            plt.savefig(savefig + "/{}_data.png".format(step), bbox_inches='tight')
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

        scores = model(mesh.detach())
        mesh = mesh.detach().numpy()
        scores = scores.detach().numpy()

        plt.grid(False)
        plt.axis('off')
        plt.quiver(mesh[:, 0], mesh[:, 1], scores[:, 0], scores[:, 1], width=0.005)
        plt.title('Estimated scores', fontsize=16)
        plt.axis('square')

        if savefig is not None:
            plt.savefig(savefig + "/{}_scores.png".format(step), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        samples = teacher.sample((1280,))
        samples = samples.detach().cpu().numpy()
        plt.scatter(samples[:, 0], samples[:, 1], s=0.1)
        plt.axis('square')
        plt.title('data samples')
        plt.xlim([left_bound, right_bound])
        plt.ylim([left_bound, right_bound])
        if savefig is not None:
            plt.savefig(savefig + "/{}_data_samples.png".format(step))
            plt.close()
        else:
            plt.show()

        samples = torch.rand(1280, 2) * (right_bound - left_bound) + left_bound
        samples = ToyRunner.langevin_dynamics(model, samples).detach().numpy()
        plt.scatter(samples[:, 0], samples[:, 1], s=0.1)
        plt.axis('square')
        plt.title('Langevin dynamics model')
        plt.xlim([left_bound, right_bound])
        plt.ylim([left_bound, right_bound])
        plt.show()

        def data_score(x):
            x = x.detach()
            x.requires_grad_(True)
            y = teacher.log_prob(x).sum()
            return autograd.grad(y, x)[0]

        scores = data_score(torch.from_numpy(mesh))
        scores = scores.detach().numpy()

        plt.axis('off')
        plt.grid(False)
        plt.quiver(mesh[:, 0], mesh[:, 1], scores[:, 0], scores[:, 1], width=0.005)
        plt.title('Data scores', fontsize=16)
        plt.axis('square')

        if savefig is not None:
            plt.savefig(savefig + "/{}_data_scores.png".format(step), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        samples = torch.rand(1280, 2) * (right_bound - left_bound) + left_bound
        samples = ToyRunner.langevin_dynamics(data_score, samples).detach().numpy()
        plt.scatter(samples[:, 0], samples[:, 1], s=0.1)
        plt.axis('square')
        plt.title('Langevin dynamics data')
        plt.xlim([left_bound, right_bound])
        plt.ylim([left_bound, right_bound])
        plt.show()

    @staticmethod
    def visualize_noise(noise_net):
        import matplotlib.pyplot as plt
        z = torch.randn(100, 2)
        # z = torch.randn(100, 1)
        with torch.no_grad():
            noise = noise_net(z)
        noise = noise.numpy()
        plt.scatter(noise[:, 0], noise[:, 1])
        plt.show()

    @staticmethod
    def visualize_iaf(noise_net):
        import matplotlib.pyplot as plt
        with torch.no_grad():
            noise, _ = noise_net.rsample(100, device='cpu')
        noise = noise.numpy()
        plt.scatter(noise[:, 0], noise[:, 1])
        plt.show()

    def fisher_information(self, energy_net, data, teacher):
        data.requires_grad_(True)
        log_pdf_model = -energy_net(data)
        model_score = autograd.grad(log_pdf_model.sum(), data)[0]
        log_pdf_actual = teacher.log_prob(data)
        actual_score = autograd.grad(log_pdf_actual.sum(), data)[0]
        return 1 / 2 * ((model_score - actual_score) ** 2).sum(1).mean(0)

    def train(self):
        hidden_units = 128
        score = nn.Sequential(
            nn.Linear(2, hidden_units),
            nn.Softplus(),
            nn.Linear(hidden_units, hidden_units),
            nn.Softplus(),
            nn.Linear(hidden_units, 2),
        )

        teacher = GMMDist(dim=2)
        optimizer = optim.Adam(score.parameters(), lr=0.001)


        sigma_begin = 1e-2
        sigma_end = 1
        num = 20
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end),
                               num))).float()

        for step in range(10000):
            samples = teacher.sample((128,))
            labels = torch.randint(0, len(sigmas), (samples.shape[0],))

            # loss, *_ = sliced_score_estimation_vr(score, samples, n_particles=1)
            # loss, *_ = sliced_score_estimation(score, samples, n_particles=10)
            loss = anneal_sliced_score_estimation_vr(score, samples, labels, sigmas, n_particles=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #logging.info('step: {}, loss: {}'.format(step, loss.item()))
            if step % 1000 == 0:
                print(step, loss.item())
        self.visualize(teacher, score, -8, 8, savefig=None)

    def plot_samples(self, samples, weights=None, range=None):
        """ Plot sample histogram in 2D """
        samples = assert_numpy(samples)
        plt.hist2d(
            samples[:, 0], 
            -samples[:, 1],
            weights=assert_numpy(weights) if weights is not None else weights,
            bins=100,
            norm=mpl.colors.LogNorm(),
            range=range
        )
        plt.show()

    def train_doublewell(self, loss = "vr", num_samples=10000):
        # Set up double well
        dim=2
        target = DoubleWellEnergy(dim, c=0.5)
        self.energy = target

        init_state = torch.Tensor([[-2, 0], [2, 0]])
        init_state = torch.cat([init_state, torch.Tensor(init_state.shape[0], dim-2).normal_()], dim=-1)
        target_sampler = GaussianMCMCSampler(target, init_state=init_state)
        data = target_sampler.sample(50000)

        hidden_units = 128
        # score = nn.Sequential(
        #     nn.Linear(2, hidden_units),
        #     nn.Softplus(),
        #     nn.Linear(hidden_units, hidden_units),
        #     nn.Softplus(),
        #     nn.Linear(hidden_units, 2),
        # )

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


    def annealed_sampling_exp(self, left_bound=-8, right_bound=8):
        sns.set(font_scale=1.3)
        sns.set_style('white')
        savefig = r'/Users/yangsong/Desktop'

        teacher = GMMDistAnneal(dim=2)
        mesh = []
        grid_size = 100
        x = np.linspace(left_bound, right_bound, grid_size)
        y = np.linspace(left_bound, right_bound, grid_size)
        for i in x:
            for j in y:
                mesh.append(np.asarray([i, j]))

        mesh = np.stack(mesh, axis=0)
        mesh = torch.from_numpy(mesh).float()

        logp_true = teacher.log_prob(mesh)
        logp_true = logp_true.view(grid_size, grid_size).exp()

        plt.grid(False)
        plt.axis('off')
        plt.imshow(np.flipud(logp_true.cpu().numpy()), cmap='inferno')

        plt.title('Data density')

        if savefig is not None:
            plt.savefig(savefig + "/density.png", bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        samples = teacher.sample((1280,))
        samples = samples.detach().cpu().numpy()
        plt.scatter(samples[:, 0], samples[:, 1], s=0.2)
        plt.axis('square')
        plt.title('i.i.d samples')
        plt.xlim([left_bound, right_bound])
        plt.ylim([left_bound, right_bound])
        if savefig is not None:
            plt.savefig(savefig + "/iid_samples.png", bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        samples = torch.rand(1280, 2) * (right_bound - left_bound) + left_bound
        samples = ToyRunner.langevin_dynamics(teacher.score, samples).detach().numpy()
        plt.scatter(samples[:, 0], samples[:, 1], s=0.2)
        plt.axis('square')
        plt.title('Langevin dynamics samples')
        plt.xlim([left_bound, right_bound])
        plt.ylim([left_bound, right_bound])
        if savefig is not None:
            plt.savefig(savefig + "/langevin_samples.png", bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        samples = torch.rand(1280, 2) * (right_bound - left_bound) + left_bound
        sigmas = np.exp(np.linspace(np.log(20), 0., 10))
        samples = ToyRunner.anneal_langevin_dynamics(teacher.score, samples, sigmas).detach().numpy()
        plt.scatter(samples[:, 0], samples[:, 1], s=0.2)
        plt.axis('square')
        plt.title('Annealed Langevin dynamics samples')
        plt.xlim([left_bound, right_bound])
        plt.ylim([left_bound, right_bound])

        if savefig is not None:
            plt.savefig(savefig + "/annealed_langevin_samples.png", bbox_inches='tight')
            plt.close()
        else:
            plt.show()

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






