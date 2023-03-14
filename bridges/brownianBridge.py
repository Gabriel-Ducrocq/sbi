import torch
import numpy as np
import matplotlib.pyplot as plt


class MixtureBrownianBridges:
    def __init__(self, T=1, sigma=1, dim_process=1):
        """

        :param T: float, Time horizon
        :param sigma: float, diffusion coefficient
        """
        self.T = T
        self.sigma = sigma
        self.sigma_torch = torch.tensor(self.sigma, dtype=torch.float32)
        self.dim_process = dim_process


    def diffusion_coeff(self, t):
        return 3*np.exp(-3*t)

    def beta_t(self, t):
        return - 5/2 * (np.exp(-2*1*t) - 1)

    def sample_marginal(self, x_0, x_T, t, T, n_sample):
        """
        Sample a variable from the marginal distribution of X_t given X_0 and X_tau, for 0 < t < tau
        :param t: float
        :return: np.array(n_sample, dim_process)
        """
        avg = x_0 + t/T*(x_T - x_0)
        std = (T - t)*t/T
        #samples = np.random.normal(size=n_sample)*std*self.sigma + avg
        samples = np.random.normal(size=n_sample)*std*self.diffusion_coeff(t)+ avg
        return np.transpose(samples)

    def compute_drift_maruyama(self, x_t, t, tau, network):
        """
        Computing the drift part of the SDE
        :param x_t:
        :param time:
        :param network:
        :return:
        """
        approximate_expectation = network.forward(torch.concat([x_t, t], dim=-1))
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
        t_old = torch.zeros((1,1), dtype=torch.float32)
        trajectories = []
        trajectories.append(x_t.detach().numpy()[0, :])
        for i, t in enumerate(times):
            drift = self.compute_drift_maruyama(x_t=x_t, t=t_old, tau=tau, network=network)
            ##Check transposition here
            x_t_new = x_t + drift * (t - t_old) + np.sqrt((t - t_old)) * torch.randn((1, self.dim_process))*self.diffusion_coeff(t)

            x_t = x_t_new
            trajectories.append(x_t.detach().numpy()[0, :])
            t_old = t

        return np.array(trajectories)




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