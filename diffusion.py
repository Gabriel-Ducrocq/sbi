import torch
import numpy as np


class Diffusion:
    def __init__(self, sqrt_gamma, b_t = None, alpha_t=None, beta_t=None, type="BM"):
        """
        Non denoising Diffusion model based on h-Transform that we will use.
        :param alpha_t: function alpha(t)
        :param beta_t:  function beta(t)
        :param tau: time horizon
        """
        self.sqrt_gamma = sqrt_gamma
        self.sqrt_gamma_torch = torch.tensor(sqrt_gamma, dtype=torch.float32)
        self.alpha_t = alpha_t
        self.beta_t = beta_t
        self.b_t = b_t
        self.dim_process = sqrt_gamma.shape[0]


    def a_t_tau(self, t, tau):
        """
        Computes the a_bm/ou function with t < tau
        :param t: float,
        :param tau: float
        :return: float
        """
        return 1.0

    def v_t_tau(self, t, tau):
        """
        Computes the v_bm/ou function with t < tau
        :param t: float
        :param tau: float
        :return: float
        """
        return self.b_t(tau) - self.b_t(t)

    def v_br(self, t, tau):
        """
        Compute the v_br for computation of the distribution of X_t given X_0 and X_tau, t < tau
        :param t: float
        :param tau: float
        :return: float
        """
        numerator = self.v_t_tau(t=0, tau=t)*self.v_t_tau(t=t, tau=tau)
        denominator = self.v_t_tau(t=0, tau=t)*self.a_t_tau(t=t, tau=tau)**2 + self.v_t_tau(t=t, tau=tau)
        return numerator/denominator

    def a_br_up(self, t, tau):
        """
        Compute the a_br_up for computation of the distribution of X_t given X_0 and X_tau, t < tau
        :param t: float
        :param tau: float
        :return: float
        """
        numerator = self.v_t_tau(t=0, tau=t)*self.a_t_tau(t=t, tau=tau)
        denominator = self.v_t_tau(t=0, tau=t)*self.a_t_tau(t=t, tau=tau)**2 + self.v_t_tau(t=t, tau=tau)
        return numerator/denominator

    def a_br_low(self, t, tau):
        """
        Compute the a_br_low for computation of the distribution of X_t given X_0 and X_tau, t < tau
        :param t: float
        :param tau: float
        :return: float
        """
        numerator = self.v_t_tau(t=t, tau=tau)*self.a_t_tau(t=0, tau=t)
        denominator = self.v_t_tau(t=0, tau=t)*self.a_t_tau(t=t, tau=tau)**2 + self.v_t_tau(t=t, tau=tau)
        return numerator/denominator

    def sample_marginal(self, x_0, x_tau, t, tau, n_sample):
        """
        Sample a variable from the marginal distribution of X_t given X_0 and X_tau, for 0 < t < tau
        :param t: float
        :return: np.array(n_sample, dim_process)
        """
        #check_avg = self.check_mean(t, tau, x_0, x_tau)
        #check_var = self.check_variance(t, tau)
        avg = x_0 * self.a_br_low(t=t, tau=tau) + x_tau * self.a_br_up(t=t, tau=tau)
        #print("Check avg:", check_avg)
        #print("Avg:", avg)

        #print("check var:", check_var)
        #print("Var:", self.v_br(t, tau))
        #assert (check_avg == avg).all(), "different means"
        #assert (check_var == self.v_br(t, tau)).all(), "different means"
        samples = np.dot(self.sqrt_gamma, np.random.normal(size=(self.dim_process, n_sample)))* np.sqrt(self.v_br(t, tau)) + avg
        #print(check_avg)
        #samples = np.random.normal(size=(self.dim_process, n_sample)) * np.sqrt(check_var) + check_avg
        return np.transpose(samples)


    def check_mean(self, t, tau, x_0, x_tau):
        return x_0 + (t/tau)*(x_tau - x_0)

    def check_variance(self, t, tau):
        return (tau - t)*t/tau

    def compute_drift_maruyama(self, x_t, observed_data, t, tau, network):
        """
        Computing the drift part of the SDE
        :param x_t:
        :param observed_data:
        :param time:
        :param network:
        :return:
        """
        approximate_expectation = network.forward(torch.concat([observed_data, x_t, t], dim = -1))
        return ((1/self.a_t_tau(t, tau))*approximate_expectation - x_t)*self.a_t_tau(t, tau)**2/self.v_t_tau(t, tau)


    def euler_maruyama(self, x_0, observed_data, times, tau, network):
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
            drift = self.compute_drift_maruyama(x_t=x_t, observed_data=observed_data, t=t_old, tau=tau, network=network)
            ##Check transposition here
            x_t_new = x_t + drift*(t - t_old) + np.sqrt((t - t_old))*torch.matmul(torch.randn((1, self.dim_process)),
                                                                torch.transpose(self.sqrt_gamma_torch, dim0=0, dim1 = 1))
            x_t = x_t_new
            trajectories.append(x_t.detach().numpy()[0, :])
            t_old = t

        return np.array(trajectories)

