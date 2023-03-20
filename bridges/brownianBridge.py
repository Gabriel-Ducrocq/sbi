import torch
import numpy as np
import matplotlib.pyplot as plt


class MixtureBrownianBridges:
    def __init__(self, times=None, dim_process=28*28, a = 3, b = 3):
        """

        :param T: float, Time horizon
        :param sigma: float, diffusion coefficient
        """
        self.T = None
        if times:
            self.T = times

        self.dim_process = dim_process
        self.a = a
        self.b = b


    def diffusion_coeff(self, t):
        return self.a*np.exp(-self.b*t)

    def beta_t(self, t):
        return - self.a**2/(2*self.b) * (np.exp(-2*self.b*t) - 1)


    def sample_marginal(self, x_0, x_T, t, T, n_sample):
        """
        Sample a variable from the marginal distribution of X_t given X_0 and X_tau, for 0 < t < tau
        :param t: float
        :return: np.array(n_sample, dim_process)
        """
        avg = x_0 * (self.beta_t(T) - self.beta_t(t))/(self.beta_t(T) - self.beta_t(0)) + x_T* self.beta_t(t)/self.beta_t(T)
        variance = (self.beta_t(T) - self.beta_t(t))/self.beta_t(T) * self.beta_t(t)
        samples = torch.randn(size=(n_sample, self.dim_process)) * np.sqrt(variance) + avg
        return samples

    def compute_drift_maruyama(self, x_t, t, tau, network):
        """
        Computing the drift part of the SDE
        :param x_t:
        :param time:
        :param network:
        :return:
        """
        input = torch.concat([x_t, t], dim=-1)
        input = input.to(dtype=torch.float32)
        approximate_expectation = network.forward(input)
        drift = (approximate_expectation - x_t)/(self.beta_t(tau) - self.beta_t(t)) * self.diffusion_coeff(t)**2
        return drift

    def euler_maruyama(self, x_0, times, tau, network):
        """

        :param x_0: torch.tensor(1, dim_process), starting point of the Euler-Maruyama scheme
        :param observed_data: torch.tensor(1, dim_data), observed data which defines the posterior distribution
        :param times: torch.tensor(N_times, 1), time discretization of the Eular-Maruyama scheme.
        :param tau: float, time horizon
        :param network: torch network, approximating the expectation
        :return: torch.tensor(1, dim_process), point approximately simulated according to the posterior distribution.
        """
        x_t = x_0
        t = torch.zeros((1,1), dtype=torch.float32)
        trajectories = []
        trajectories.append(x_t.detach().numpy()[0, :])
        for i, t_new in enumerate(times):
            drift = self.compute_drift_maruyama(x_t=x_t, t=t, tau=tau, network=network)
            ##Check transposition here
            x_t_new = x_t + drift * (t_new - t) + np.sqrt((t_new - t)) * torch.randn((1, self.dim_process))*self.diffusion_coeff(t)

            x_t = x_t_new
            trajectories.append(x_t.detach().numpy()[0, :])
            t = t_new

        return np.array(trajectories), x_t




"""
mu1 = 0
mu2 = 0
def likelihood(mu1, mu2):
    mu = mu1
    if np.random.uniform() < 0.5:
        mu = mu2

    return np.random.normal(size=(1,)) + mu

dataset_sampled = np.array([likelihood(mu1, mu2) for _ in range(1000)])

mixt = MixtureBrownianBridges(sigma=15)
times = np.linspace(0, 1, 100)
traj = []
for i, data in enumerate(dataset_sampled):
    x_0 = np.random.normal()
    print(i)
    traj.append(mixt.sample_marginal(x_0, data, times, 1, 100))

for i, tr in enumerate(traj):
    print(i)
    plt.plot(tr, alpha=0.5)

plt.show()
"""