import numpy as np
import torch

class PhysicalLangevinDynamicsMonoatomic:

    def __init__(self, init_positions, masses, dt, temperature):
        """
        Parameters
        ----------
        masses : masses in amu
        timestep: the time step in femtoseconds
        temperature: the temperature in Kelvin
        """

        self.positions = init_positions
        self.v = torch.zeros_like(init_positions)

        self.dt = dt # fs
        self.temperature = temperature

        fr = 5e-1

        self.masses = masses

        # self.c1 = self.dt / 2. - self.dt * self.dt * fr / 8.
        # self.c3 = np.sqrt(self.dt) * sigma / 2. - self.dt**1.5 * fr * sigma / 8.
        # self.c5 = self.dt**1.5 * sigma / (2 * np.sqrt(3))
        # self.c4 = fr / 2. * self.c5

        # self.c3 = self.c3.view(-1,1)
        # self.c4 = self.c4.view(-1,1)
        # self.c5 = self.c5.view(-1,1)

        self.fr = fr # 1 / fs

        self.boltzmann_constant = 1.380649e-23 # joules / K

        self.eV_to_joules = 1/6.24e18
        self.m_to_A = 1e10
        self.amu_to_kg = 1.66054e-27
        self.s_to_fs = 1e15
        
        self.sigma = 2 * self.boltzmann_constant * self.temperature * self.fr * self.dt / self.masses # joules / amu
        self.sigma = self.sigma / (self.s_to_fs**2) # kg * m**2 / (fs**2 * amu)
        self.sigma = np.sqrt(self.sigma / self.amu_to_kg) # m / fs

        self.c1 = self.eV_to_joules * (self.m_to_A**2) / (self.amu_to_kg * (self.s_to_fs**2))


    def step(self, force_net):

        xi = torch.randn_like(self.positions)
        eta = torch.randn_like(self.positions)

        rnd_pos = self.c5 * eta
        rnd_vel = self.c3 * xi - self.c4 * eta

        _, forces = force_net(self.positions)
        forces = forces.detach()

        self.v += (self.c1 * forces / self.masses - self.c2 * self.v +
                   rnd_vel)

        init_positions = torch.clone(self.positions)
        self.positions = self.positions + self.dt * self.v + rnd_pos


        self.v = (self.positions - init_positions - rnd_pos) / self.dt
        _, forces = force_net(self.positions)
        forces = forces.detach()

        # Update the velocities
        self.v += (self.c1 * forces / self.masses - self.c2 * self.v +
                   rnd_vel)


    def true_step(self, force_generator):

        eta = torch.randn_like(self.positions)
        rnd_vel = self.sigma.view(-1,1) * eta
        
        forces = force_generator.force(self.positions) # eV / A

        self.v += (self.c1 * forces * self.dt / self.masses.view(-1,1) - self.fr * self.v * self.dt + rnd_vel) # A / fs

        self.positions = self.positions + self.dt * self.v